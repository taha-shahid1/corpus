from __future__ import annotations


def __getattr__(name: str):
    if name == "ingest":
        from corpus.ingestion.pipeline import ingest

        return ingest
    if name == "ingest_url":
        from corpus.ingestion.pipeline import ingest_url

        return ingest_url
    raise AttributeError(f"module 'corpus' has no attribute {name!r}")


__all__ = ["ingest", "ingest_url"]
