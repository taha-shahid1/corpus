from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf
from langchain_core.documents import Document

from corpus.ingestion.loaders.base import Loader

_MIN_PAGE_CHARS = 50  # skip near-blank pages (page numbers, section dividers, etc.)


class PDFLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        # Absolute path keeps the storage key stable across cwd changes
        self.source = str(Path(path).resolve()) if path else path

    def load(self) -> list[Document]:
        src = Path(self.source)
        if not src.exists():
            raise FileNotFoundError(f"PDF not found: {self.source!r}")

        doc = fitz.open(str(src))
        pages: list[Document] = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) >= _MIN_PAGE_CHARS:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": self.source, "page": page_num},
                    )
                )

        doc.close()

        if not pages:
            raise ValueError(f"No readable text in {self.source!r}")

        return pages


assert isinstance(PDFLoader(""), Loader)
