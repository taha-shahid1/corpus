from __future__ import annotations

import logging

import lancedb

from corpus.config import LANCEDB_TABLE, LANCEDB_URI
from corpus.ingestion.loaders.base import Loader
from corpus.ingestion.loaders.pdf import PDFLoader
from corpus.ingestion.loaders.web import WebLoader
from corpus.retrieval.retriever import build_retriever
from corpus.storage import is_ingested, mark_ingested

logger = logging.getLogger(__name__)


def ingest(loader: Loader):
    """Ingest documents from loader into the Corpus stores.

    Raises ValueError if the loader produces no documents.
    """
    if is_ingested(loader.source):
        logger.info("%s already ingested, skipping", loader.source)
        return build_retriever()

    documents = loader.load()
    if not documents:
        raise ValueError(f"{loader!r} returned no documents")

    logger.info("Loaded %d document(s)", len(documents))

    retriever = build_retriever()
    retriever.add_documents(documents)

    # Rebuild FTS index so new documents are searchable by keyword
    db = lancedb.connect(LANCEDB_URI)
    try:
        db.open_table(LANCEDB_TABLE).create_fts_index("text", replace=True)
        logger.info("FTS index rebuilt")
    except Exception as e:
        logger.warning("FTS index creation failed: %s", e)

    mark_ingested(loader.source, len(documents))
    logger.info("Ingestion complete")

    return retriever


def ingest_url(url: str):
    return ingest(WebLoader(url))


def ingest_pdf(path: str):
    return ingest(PDFLoader(path))
