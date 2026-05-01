from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from corpus.config import RERANKER_DEVICE, RERANKER_MODEL, RERANKER_TOP_K

_model: CrossEncoder | None = None


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        device = RERANKER_DEVICE or _detect_device()
        _model = CrossEncoder(RERANKER_MODEL, device=device)
    return _model


def warmup() -> None:
    """Pre-load the cross-encoder so the first query doesn't pay the cold-start cost."""
    _get_model()


def rerank(query: str, docs: list[Document], top_k: int = RERANKER_TOP_K) -> list[Document]:
    scores = _get_model().predict([(query, doc.page_content) for doc in docs])
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
