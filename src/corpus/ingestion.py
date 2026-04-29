from __future__ import annotations

import logging
from pathlib import Path

import lancedb
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import LanceDB
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus.config import (
    CHILD_CHUNK_SIZE,
    DOCSTORE_PATH,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    LANCEDB_TABLE,
    LANCEDB_URI,
)
from corpus.loaders.base import Loader
from corpus.loaders.web import WebLoader

logger = logging.getLogger(__name__)

class _HybridLanceDB(LanceDB):
    """LanceDB vectorstore that uses hybrid search (vector + BM25) by default."""

    def similarity_search(self, query: str, k: int = 4, **kwargs):
        kwargs.setdefault("query_type", "hybrid")
        return super().similarity_search(query, k=k, **kwargs)

class _SemanticTextSplitter(RecursiveCharacterTextSplitter):
    """Wraps SemanticChunker as a TextSplitter.

    langchain_classic's ParentDocumentRetriever requires parent_splitter to be a
    TextSplitter subclass, but SemanticChunker only extends BaseDocumentTransformer.
    """

    def __init__(self, chunker: SemanticChunker) -> None:
        super().__init__(chunk_size=10_000_000)
        self._chunker = chunker

    def split_text(self, text: str) -> list[str]:
        return [doc.page_content for doc in self._chunker.create_documents([text])]


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_retriever(embeddings: HuggingFaceEmbeddings) -> ParentDocumentRetriever:
    vectorstore = _HybridLanceDB(
        connection=lancedb.connect(LANCEDB_URI),
        embedding=embeddings,
        table_name=LANCEDB_TABLE,
    )

    Path(DOCSTORE_PATH).mkdir(parents=True, exist_ok=True)
    docstore = create_kv_docstore(LocalFileStore(str(DOCSTORE_PATH)))

    parent_splitter = _SemanticTextSplitter(
        SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
    )

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE),
        parent_splitter=parent_splitter,
    )


def ingest(loader: Loader) -> ParentDocumentRetriever:
    """Ingest documents from loader into the Corpus stores.

    Raises ValueError if the loader produces no documents.
    """
    documents = loader.load()
    if not documents:
        raise ValueError(f"{loader!r} returned no documents")

    logger.info("Loaded %d document(s)", len(documents))

    embeddings = _build_embeddings()
    retriever = _build_retriever(embeddings)

    retriever.add_documents(documents)
    # Rebuild FTS index so new documents are searchable by keyword
    db = lancedb.connect(LANCEDB_URI)
    try:
        db.open_table(LANCEDB_TABLE).create_fts_index("text", replace=True)
        logger.info("FTS index rebuilt")
    except Exception as e:
        logger.warning("FTS index creation failed: %s", e)
    logger.info("Ingestion complete")

    return retriever


def get_retriever() -> ParentDocumentRetriever:
    """Return a retriever connected to the existing stores, for querying."""
    return _build_retriever(_build_embeddings())


def ingest_url(url: str) -> ParentDocumentRetriever:
    return ingest(WebLoader(url))
