from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from corpus.ingestion.loaders.base import Loader

_SECTION_CATEGORIES = {"Title"}
_SKIP_CATEGORIES = {"Footer", "Header"}  # page-repeated noise
_MIN_GROUP_CHARS = 200
_MIN_GROUPS = 3


class PDFLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        # Absolute path keeps the storage key stable across cwd changes.
        self.source = str(Path(path).resolve()) if path else path

    def load(self) -> list[Document]:
        from unstructured.partition.pdf import partition_pdf

        src = Path(self.source)
        if not src.exists():
            raise FileNotFoundError(f"PDF not found: {self.source!r}")

        elements = partition_pdf(
            filename=str(src),
            strategy="fast",
            include_page_breaks=True,
        )

        if not elements:
            raise ValueError(f"No content extracted from {self.source!r}")

        groups = self._group_by_section(elements)
        groups = [g for g in groups if len(g.page_content) >= _MIN_GROUP_CHARS]

        if len(groups) >= _MIN_GROUPS:
            return groups

        combined = "\n\n".join(
            str(el)
            for el in elements
            if el.category not in {"PageBreak", "Footer"} and str(el).strip()
        )
        if not combined.strip():
            raise ValueError(f"No readable text in {self.source!r}")
        return [Document(page_content=combined, metadata={"source": self.source})]

    def _group_by_section(self, elements: list) -> list[Document]:
        groups: list[Document] = []
        buffer: list[str] = []
        current_page: int | None = None
        group_start_page: int | None = None

        def flush() -> None:
            text = "\n\n".join(p for p in buffer if p.strip())
            if text:
                meta: dict = {"source": self.source}
                if group_start_page is not None:
                    meta["page"] = group_start_page
                groups.append(Document(page_content=text, metadata=meta))
            buffer.clear()

        for el in elements:
            cat = el.category
            page = _element_page(el)

            if page is not None:
                current_page = page

            if cat == "PageBreak":
                continue

            if cat in _SKIP_CATEGORIES:
                continue

            if cat in _SECTION_CATEGORIES:
                flush()
                group_start_page = current_page

            text = str(el).strip()
            if text:
                if not buffer:
                    group_start_page = current_page
                buffer.append(text)

        flush()
        return groups


def _element_page(element) -> int | None:
    try:
        return element.metadata.page_number
    except AttributeError:
        return None


assert isinstance(PDFLoader(""), Loader)
