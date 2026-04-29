from corpus.ingestion.loaders.base import Loader
from corpus.ingestion.loaders.web import WebLoader
from corpus.ingestion.pipeline import ingest, ingest_url

__all__ = ["Loader", "WebLoader", "ingest", "ingest_url"]
