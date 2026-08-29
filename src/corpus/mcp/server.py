from __future__ import annotations

import logging
from pathlib import Path

import anyio
from mcp.server.mcpserver import Context, MCPServer

from corpus.config import RERANKER_TOP_K
from corpus.ingestion import ingest_md, ingest_pdf, ingest_url
from corpus.retrieval.reranker import rerank
from corpus.retrieval.reranker import warmup as warmup_reranker
from corpus.retrieval.retriever import get_retriever
from corpus.store.ingestion import is_ingested

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "Corpus is the user's personal knowledge base: ingested PDFs, Markdown notes, and web "
    "articles. Call `search` to find relevant passages before answering questions about the "
    "user's own documents or notes — not for general knowledge you already have. Call `add` "
    "to ingest a new local PDF/Markdown file or URL so future `search` calls can find it."
)


def _display_source(src: str) -> str:
    """Short display name: filename for local paths, full string for URLs."""
    if src.startswith(("http://", "https://")):
        return src
    return src.rsplit("/", 1)[-1] or src


def _resolve_source(source: str) -> tuple[str, str, str | None]:
    """Return (ingest_key, display_name, ext). ext is None for URLs."""
    if source.startswith(("http://", "https://")):
        return source, source, None
    resolved = str(Path(source).resolve())
    return resolved, Path(resolved).name, Path(resolved).suffix.lower()


def warmup() -> None:
    """Eagerly load the embedding and reranker models so the first tool call isn't slow."""
    get_retriever()
    warmup_reranker()


def build_server() -> MCPServer:
    server = MCPServer(name="corpus", instructions=_INSTRUCTIONS)

    @server.tool()
    def search(query: str, k: int = RERANKER_TOP_K) -> list[dict]:
        """Search the user's personal knowledge base for passages relevant to *query*.

        Runs hybrid (vector + keyword) retrieval over ingested PDFs, Markdown notes, and
        web articles, then reranks candidates with a cross-encoder. Returns up to *k*
        passages ordered by relevance, each with its source, page (if any), and a
        relevance score — higher is more relevant, scores below ~0 are likely irrelevant.
        """
        docs = get_retriever().invoke(query)
        if not docs:
            return []
        return [
            {
                "content": doc.page_content,
                "source": _display_source(doc.metadata.get("source", "")),
                "page": doc.metadata.get("page"),
                "score": score,
            }
            for doc, score in rerank(query, docs, top_k=k)
        ]

    @server.tool()
    async def add(source: str, ctx: Context) -> dict:
        """Ingest a local PDF/Markdown file or a URL into the knowledge base.

        *source* is a filesystem path or an http(s) URL. A source that's already been
        ingested is skipped, not re-ingested. Ingestion parses, embeds, and indexes the
        content, so it can take a while for large files — progress is reported as it runs.
        """
        ingest_key, display_name, ext = _resolve_source(source)

        if ext is not None and ext not in (".pdf", ".md"):
            return {
                "status": "error",
                "source": display_name,
                "message": f"unsupported file type {ext!r} (supported: .pdf, .md)",
            }

        if is_ingested(ingest_key):
            return {"status": "already_indexed", "source": display_name}

        def on_status(msg: str) -> None:
            anyio.from_thread.run(ctx.report_progress, 0, None, msg)

        def _do_ingest() -> None:
            if ext is None:
                ingest_url(source, on_status=on_status)
            elif ext == ".pdf":
                ingest_pdf(ingest_key, on_status=on_status)
            else:
                ingest_md(ingest_key, on_status=on_status)

        try:
            await anyio.to_thread.run_sync(_do_ingest)
        except (ValueError, FileNotFoundError) as exc:
            return {"status": "error", "source": display_name, "message": str(exc)}

        return {"status": "ingested", "source": display_name}

    return server
