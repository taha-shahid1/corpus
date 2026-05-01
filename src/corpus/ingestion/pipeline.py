from __future__ import annotations

import logging
import threading

from corpus.config import LANCEDB_TABLE
from corpus.ingestion.loaders.base import Loader
from corpus.ingestion.loaders.md import MarkdownLoader
from corpus.ingestion.loaders.pdf import PDFLoader
from corpus.ingestion.loaders.web import WebLoader
from corpus.retrieval.retriever import get_lancedb, get_retriever
from corpus.storage import is_ingested, mark_ingested

logger = logging.getLogger(__name__)

# Serialise add_documents + FTS rebuild so concurrent watcher threads don't corrupt the index.
_write_lock = threading.Lock()


def remove_source(source: str) -> None:
    """Delete all child embeddings and parent chunks for *source*.

    Child chunks are stored in LanceDB; parent chunks are stored in the docstore.
    Metadata in LanceDB is nested under a ``metadata`` struct, so filters must use
    ``metadata.source`` rather than a top-level ``source`` column.
    """
    try:
        tbl = get_lancedb().open_table(LANCEDB_TABLE)
        escaped = source.replace("'", "''")

        # Collect parent IDs from child chunks before deleting them.
        rows = (
            tbl.search()
            .where(f"metadata.source = '{escaped}'", prefilter=True)
            .select(["metadata"])
            .to_arrow()
        )
        parent_ids = list({row["doc_id"].as_py() for row in rows["metadata"]})

        tbl.delete(f"metadata.source = '{escaped}'")

        if parent_ids:
            get_retriever().docstore.mdelete(parent_ids)
            logger.info(
                "Removed %d parent chunks and all child embeddings for %s", len(parent_ids), source
            )
        else:
            logger.info("Removed embeddings for %s (no parent chunks found)", source)
    except Exception as e:
        logger.warning("Could not fully remove %s: %s", source, e)


# Backward-compatible alias.
remove_source_embeddings = remove_source


def rebuild_fts_index() -> None:
    """Rebuild the LanceDB full-text search index. Call once after a batch of ingestions."""
    try:
        get_lancedb().open_table(LANCEDB_TABLE).create_fts_index("text", replace=True)
        logger.info("FTS index rebuilt")
    except Exception as e:
        logger.warning("FTS index creation failed: %s", e)


def ingest(
    loader: Loader, *, force: bool = False, file_hash: str | None = None, rebuild_index: bool = True
):
    """Ingest documents from *loader* into the Corpus stores.

    Pass ``force=True`` to re-ingest a source that is already registered
    (e.g. when the file has changed on disk).

    Pass ``rebuild_index=False`` when ingesting many files in a batch; call
    ``rebuild_fts_index()`` once after the batch completes instead.

    Raises ValueError if the loader produces no documents.
    """
    if not force and is_ingested(loader.source):
        logger.info("%s already ingested, skipping", loader.source)
        return get_retriever()

    documents = loader.load()
    if not documents:
        raise ValueError(f"{loader!r} returned no documents")

    logger.info("Loaded %d document(s)", len(documents))

    # Retrieve the cached singleton outside the lock — it's a no-op after first call.
    retriever = get_retriever()

    with _write_lock:
        retriever.add_documents(documents)
        if rebuild_index:
            rebuild_fts_index()

    mark_ingested(loader.source, len(documents), file_hash=file_hash)
    logger.info("Ingestion complete")

    return retriever


def ingest_url(url: str):
    return ingest(WebLoader(url))


def ingest_pdf(
    path: str, *, force: bool = False, file_hash: str | None = None, rebuild_index: bool = True
):
    return ingest(PDFLoader(path), force=force, file_hash=file_hash, rebuild_index=rebuild_index)


def ingest_md(
    path: str, *, force: bool = False, file_hash: str | None = None, rebuild_index: bool = True
):
    return ingest(
        MarkdownLoader(path), force=force, file_hash=file_hash, rebuild_index=rebuild_index
    )
