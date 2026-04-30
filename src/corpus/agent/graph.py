from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from corpus.agent.llm import LLMProvider
from corpus.agent.nodes import (
    generate_node,
    grade_node,
    plan_node,
    retrieve_node,
    rewrite_node,
    route_after_grade,
)
from corpus.agent.state import AgentState
from corpus.retrieval.retriever import build_retriever

logger = logging.getLogger(__name__)


def build_graph(
    llm: LLMProvider | None = None,
    retriever: Runnable[str, list[Document]] | None = None,
) -> CompiledStateGraph:
    """Assemble and compile the RAG agent graph."""
    if llm is None:
        from corpus.agent.llm import default_provider

        llm = default_provider()

    if retriever is None:
        retriever = build_retriever()

    _plan = plan_node(llm)
    _retrieve = retrieve_node(retriever)
    _grade = grade_node(llm)
    _rewrite = rewrite_node(llm)
    _generate = generate_node(llm)

    graph = StateGraph(AgentState)

    graph.add_node("plan", _plan)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("grade", _grade)
    graph.add_node("rewrite", _rewrite)
    graph.add_node("generate", _generate)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("rewrite", "plan")
    graph.add_edge("generate", END)

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    return graph.compile()