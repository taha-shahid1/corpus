from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from corpus.config import RERANKER_DEVICE, RERANKER_MODEL, RERANKER_TOP_K

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL, device=RERANKER_DEVICE)
    return _model


def rerank(query: str, docs: list[Document], top_k: int = RERANKER_TOP_K) -> list[Document]:
    scores = _get_model().predict([(query, doc.page_content) for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
