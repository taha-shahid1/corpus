from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State shared by the RAG agent graph."""

    query: str
    original_query: str
    route_type: str  # "rag" | "direct"
    sub_questions: list[str]
    docs: list[Document]
    top_rerank_score: float
    loop_count: int
    answer: str
    messages: Annotated[list[BaseMessage], add_messages]
