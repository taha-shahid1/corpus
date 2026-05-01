from __future__ import annotations

import logging
import threading
from pathlib import Path

import lancedb
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus.config import (
    CHILD_CHUNK_SIZE,
    DOCSTORE_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    LANCEDB_TABLE,
    LANCEDB_URI,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()
_embeddings: HuggingFaceEmbeddings | None = None
_lancedb_conn: lancedb.LanceDBConnection | None = None
_retriever: ParentDocumentRetriever | None = None


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _HybridLanceDB(LanceDB):
    """LanceDB vectorstore that uses hybrid search (vector + BM25) by default.

    Falls back to vector-only search if no FTS index exists yet (empty corpus).
    The FTS index is managed externally by ``rebuild_fts_index()`` after ingestion.
    """

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        if self._embedding is None:
            raise ValueError("An embedding function is required.")
        embedding = self._embedding.embed_query(query)
        tbl = self.get_table()
        try:
            results = (
                tbl.search(query_type="hybrid", vector_column_name=self._vector_key)
                .vector(embedding)
                .text(query)
                .limit(k)
                .to_arrow()
            )
        except Exception:
            logger.warning("Hybrid search unavailable, falling back to vector-only search")
            results = tbl.search(embedding, vector_column_name=self._vector_key).limit(k).to_arrow()
        return self.results_to_docs(results, score=False)


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        with _init_lock:
            if _embeddings is None:
                device = EMBEDDING_DEVICE or _detect_device()
                logger.info("Embedding device: %s", device)
                _embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": device, "trust_remote_code": True},
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": EMBEDDING_BATCH_SIZE,
                    },
                )
    return _embeddings


def get_lancedb() -> lancedb.LanceDBConnection:
    """Return the shared LanceDB connection, creating it on first call."""
    global _lancedb_conn
    if _lancedb_conn is None:
        with _init_lock:
            if _lancedb_conn is None:
                _lancedb_conn = lancedb.connect(LANCEDB_URI)
    return _lancedb_conn


def get_retriever() -> ParentDocumentRetriever:
    """Return the shared retriever, creating it on first call."""
    global _retriever
    if _retriever is None:
        with _init_lock:
            if _retriever is None:
                _retriever = _build_retriever()
    return _retriever


def _build_retriever() -> ParentDocumentRetriever:
    embeddings = _get_embeddings()

    vectorstore = _HybridLanceDB(
        connection=get_lancedb(),
        embedding=embeddings,
        table_name=LANCEDB_TABLE,
    )

    Path(DOCSTORE_PATH).mkdir(parents=True, exist_ok=True)
    docstore = create_kv_docstore(LocalFileStore(str(DOCSTORE_PATH)))

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE),
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
        ),
    )


# Backward-compatible alias — callers that held a reference to build_retriever still work.
build_retriever = get_retriever
