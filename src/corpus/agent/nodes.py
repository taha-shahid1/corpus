from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from corpus.agent.llm import LLMProvider
from corpus.agent.state import AgentState
from corpus.config import AGENT_MAX_LOOPS
from corpus.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


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
    relevant: bool = Field(
        description="True if the document contains information useful for answering the query."
    )


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
        logger.debug("GRADE: evaluating %d docs", len(docs))

        relevant: list[Document] = []
        for doc in docs:
            result: GradeOutput = structured.invoke(
                [
                    HumanMessage(
                        content=(
                            "You are a relevance grader. "
                            "Given a user query and a retrieved document, decide if the document "
                            "is relevant to answering the query.\n\n"
                            f"Query: {query}\n\n"
                            f"Document:\n{doc.page_content}"
                        )
                    )
                ]
            )
            if result.relevant:
                relevant.append(doc)

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

        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"doc_{i}")
            context_parts.append(f"[{i + 1}] (source: {source})\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        result = llm.strong.invoke(
            [
                HumanMessage(
                    content=(
                        "You are a helpful assistant. Answer the user's question using only the "
                        "provided context. Cite sources inline using [1], [2], etc. matching the "
                        "numbered context blocks. If the context lacks enough information, "
                        "say so.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {query}"
                    )
                )
            ]
        )

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