from __future__ import annotations

import logging
from pathlib import Path

import lancedb
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
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
)

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


class _HybridLanceDB(LanceDB):
    """LanceDB vectorstore that uses hybrid search (vector + BM25) by default."""

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        if self._embedding is None:
            raise ValueError("An embedding function is required.")
        tbl = self.get_table()
        if self._fts_index is None:
            self._fts_index = tbl.create_fts_index(self._text_key, replace=True)
        embedding = self._embedding.embed_query(query)
        results = (
            tbl.search(query_type="hybrid", vector_column_name=self._vector_key)
            .vector(embedding)
            .text(query)
            .limit(k)
            .to_arrow()
        )
        return self.results_to_docs(results, score=False)


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


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": EMBEDDING_BATCH_SIZE},
        )
    return _embeddings


def build_retriever() -> ParentDocumentRetriever:
    embeddings = _get_embeddings()

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
