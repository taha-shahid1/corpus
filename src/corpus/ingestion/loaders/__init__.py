from corpus.ingestion.loaders.base import Loader
from corpus.ingestion.loaders.pdf import PDFLoader
from corpus.ingestion.loaders.web import WebLoader

__all__ = ["Loader", "PDFLoader", "WebLoader"]
