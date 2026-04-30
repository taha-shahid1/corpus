from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from corpus.ingestion.loaders.base import Loader

_SECTION_CATEGORIES = {"Title"}
_SKIP_CATEGORIES = {"Footer", "Header"}
_MIN_GROUP_CHARS = 200
_MIN_GROUPS = 3


class MarkdownLoader:
    def __init__(self, path: str) -> None:
        self.path = path
        # Absolute path keeps the storage key stable across cwd changes.
        self.source = str(Path(path).resolve()) if path else path

    def load(self) -> list[Document]:
        from unstructured.partition.md import partition_md

        src = Path(self.source)
        if not src.exists():
            raise FileNotFoundError(f"Markdown file not found: {self.source!r}")

        elements = partition_md(filename=str(src))

        if not elements:
            raise ValueError(f"No content extracted from {self.source!r}")

        groups = self._group_by_section(elements)
        groups = [g for g in groups if len(g.page_content) >= _MIN_GROUP_CHARS]

        if len(groups) >= _MIN_GROUPS:
            return groups

        combined = "\n\n".join(str(el) for el in elements if str(el).strip())
        if not combined.strip():
            raise ValueError(f"No readable text in {self.source!r}")
        return [Document(page_content=combined, metadata={"source": self.source})]

    def _group_by_section(self, elements: list) -> list[Document]:
        groups: list[Document] = []
        buffer: list[str] = []

        def flush() -> None:
            text = "\n\n".join(p for p in buffer if p.strip())
            if text:
                groups.append(Document(page_content=text, metadata={"source": self.source}))
            buffer.clear()

        for el in elements:
            if el.category in _SKIP_CATEGORIES:
                continue
            if el.category in _SECTION_CATEGORIES:
                flush()
            text = str(el).strip()
            if text:
                buffer.append(text)

        flush()
        return groups


assert isinstance(MarkdownLoader(""), Loader)
