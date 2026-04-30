from __future__ import annotations

import logging
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from corpus.agent.llm import LLMProvider
from corpus.agent.state import AgentState
from corpus.config import AGENT_MAX_LOOPS
from corpus.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


class RouteOutput(BaseModel):
    route: Literal["rag", "direct"] = Field(
        description=(
            "rag: the query is a factual or knowledge question that may be answered by "
            "searching a personal knowledge base (papers, articles, docs). "
            "direct: the query is conversational, a greeting, a thank-you, an opinion "
            "request, or clearly not a knowledge lookup."
        )
    )


class PlanOutput(BaseModel):
    sub_questions: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=3,
            description=(
                "1 to 3 focused sub-questions that together cover the user query. "
                "Use exactly 1 sub-question for simple/factual queries."
            ),
        ),
    ]


class GradeOutput(BaseModel):
    relevant_indices: Annotated[
        list[int],
        Field(
            description=(
                "Zero-based indices of documents that contain information useful for "
                "answering the query. Return an empty list if none are relevant."
            )
        ),
    ]


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])

    return ("\n".join(parts) if parts else str(content)).strip()


def plan_node(llm: LLMProvider):
    structured = llm.fast.with_structured_output(PlanOutput)

    def _run(state: AgentState) -> dict:
        query = state["query"]
        logger.debug("PLAN: %s", query)

        result: PlanOutput = structured.invoke(
            [
                HumanMessage(
                    content=(
                        "You are a query planning assistant. "
                        "Decompose the following user query into 1-3 focused sub-questions. "
                        "Use a single sub-question for simple or factual queries. "
                        "Output nothing except the JSON.\n\n"
                        f"Query: {query}"
                    )
                )
            ]
        )

        logger.debug("PLAN sub_questions: %s", result.sub_questions)
        updates: dict = {"sub_questions": result.sub_questions}
        if not state.get("original_query"):
            updates["original_query"] = query
        return updates

    return _run


def retrieve_node(retriever: Runnable[str, list[Document]]):
    def _run(state: AgentState) -> dict:
        query = state["query"]
        sub_questions = state.get("sub_questions") or [query]
        logger.debug("RETRIEVE for %d sub-questions", len(sub_questions))

        seen: set[str] = set()
        all_docs: list[Document] = []

        for sq in sub_questions:
            for doc in retriever.invoke(sq):
                key = doc.page_content.strip()
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        if not all_docs:
            return {"docs": []}

        ranked = rerank(query, all_docs)
        logger.debug(
            "RETRIEVE: %d unique docs -> top %d after rerank",
            len(all_docs),
            len(ranked),
        )
        return {"docs": ranked}

    return _run


def grade_node(llm: LLMProvider):
    structured = llm.fast.with_structured_output(GradeOutput)

    def _run(state: AgentState) -> dict:
        query = state["query"]
        docs = state["docs"]
        logger.debug("GRADE: evaluating %d docs in one batch call", len(docs))

        if not docs:
            return {"docs": []}

        numbered = "\n\n".join(
            f"[{i}] {doc.page_content[:600]}" for i, doc in enumerate(docs)
        )
        result: GradeOutput = structured.invoke(
            [
                HumanMessage(
                    content=(
                        "You are a relevance grader. "
                        "Given a user query and a numbered list of retrieved documents, "
                        "return the zero-based indices of every document that contains "
                        "information useful for answering the query. "
                        "Return an empty list if none are relevant.\n\n"
                        f"Query: {query}\n\n"
                        f"Documents:\n{numbered}"
                    )
                )
            ]
        )

        relevant = [docs[i] for i in result.relevant_indices if 0 <= i < len(docs)]
        logger.debug("GRADE: %d/%d docs kept", len(relevant), len(docs))
        return {"docs": relevant}

    return _run


def rewrite_node(llm: LLMProvider):
    def _run(state: AgentState) -> dict:
        original = state.get("original_query") or state["query"]
        current = state["query"]
        loop = state.get("loop_count", 0)
        logger.debug("REWRITE (loop %d) from original: %s", loop, original)

        result = llm.fast.invoke(
            [
                HumanMessage(
                    content=(
                        "The current retrieval query failed to find relevant documents. "
                        "Rewrite the original user question as a different, concise search query. "
                        "Use alternative terminology or a narrower focus. "
                        "Output only the rewritten query, with no explanation.\n\n"
                        f"Original question: {original}\n"
                        f"Current retrieval query: {current}"
                    )
                )
            ]
        )

        new_query = _message_text(result)
        logger.debug("REWRITE result: %s", new_query)
        return {
            "query": new_query,
            "loop_count": loop + 1,
        }

    return _run


def generate_node(llm: LLMProvider):
    def _run(state: AgentState) -> dict:
        query = state.get("original_query") or state["query"]
        docs = state["docs"]
        logger.debug("GENERATE: synthesising from %d docs", len(docs))

        if docs:
            context = "\n\n---\n\n".join(
                f"[{i + 1}] (source: {doc.metadata.get('source', f'doc_{i}')})\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
            prompt = (
                "You are a helpful assistant. Answer the user's question using only the "
                "provided context. Cite sources inline using [1], [2], etc. matching the "
                "numbered context blocks. If the context lacks enough information, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        else:
            prompt = (
                "You are a helpful assistant. The knowledge base contains no relevant documents "
                "for this query. Tell the user you don't have relevant information on this topic "
                "and do NOT cite any sources or use bracket notation like [1].\n\n"
                f"Question: {query}"
            )

        result = llm.strong.invoke([HumanMessage(content=prompt)])

        answer = _message_text(result)
        logger.debug("GENERATE complete (%d chars)", len(answer))

        return {
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
        }

    return _run


def route_after_grade(state: AgentState) -> str:
    has_docs = bool(state.get("docs"))
    over_limit = state.get("loop_count", 0) >= AGENT_MAX_LOOPS

    if has_docs or over_limit:
        return "generate"
    return "rewrite"


def route_node(llm: LLMProvider):
    structured = llm.fast.with_structured_output(RouteOutput)

    def _run(state: AgentState) -> dict:
        query = state["query"]
        logger.debug("ROUTE: classifying query")

        result: RouteOutput = structured.invoke(
            [
                HumanMessage(
                    content=(
                        "Classify the following user query as either 'rag' or 'direct'.\n\n"
                        "rag — a factual or knowledge question that could be answered by "
                        "searching a personal knowledge base of papers, articles, or docs.\n"
                        "direct — conversational, a greeting, a thank-you, an opinion request, "
                        "or anything clearly not a knowledge lookup.\n\n"
                        "Output nothing except the JSON.\n\n"
                        f"Query: {query}"
                    )
                )
            ]
        )

        logger.debug("ROUTE decision: %s", result.route)
        return {"route_type": result.route}

    return _run


def respond_node(llm: LLMProvider):
    """Direct response for conversational queries — bypasses the full RAG pipeline."""

    def _run(state: AgentState) -> dict:
        query = state["query"]
        logger.debug("RESPOND: direct conversational reply")

        result = llm.strong.invoke(
            [
                HumanMessage(
                    content=(
                        "You are a helpful assistant. Respond naturally to the user's message. "
                        "Be concise and friendly. Do not mention a knowledge base.\n\n"
                        f"Message: {query}"
                    )
                )
            ]
        )

        answer = _message_text(result)
        logger.debug("RESPOND complete (%d chars)", len(answer))
        return {
            "answer": answer,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
        }

    return _run


def route_after_classify(state: AgentState) -> str:
    return "plan" if state.get("route_type") == "rag" else "respond"