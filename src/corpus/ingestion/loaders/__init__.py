from corpus.ingestion.loaders.base import Loader
from corpus.ingestion.loaders.md import MarkdownLoader
from corpus.ingestion.loaders.pdf import PDFLoader
from corpus.ingestion.loaders.web import WebLoader

__all__ = ["Loader", "MarkdownLoader", "PDFLoader", "WebLoader"]
