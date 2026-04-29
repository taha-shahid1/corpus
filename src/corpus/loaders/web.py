from langchain_core.documents import Document
from unstructured.partition.html import partition_html

from corpus.loaders.base import Loader

_HEADER_CATEGORIES = {"Title", "Header"}
_MIN_GROUP_CHARS = 200
_MIN_GROUPS = 3


class WebLoader:
    def __init__(self, url: str) -> None:
        self.url = url
        self.source = url

    def load(self) -> list[Document]:
        elements = partition_html(url=self.url)
        if not elements:
            raise ValueError(f"No content extracted from {self.url!r}")

        groups = self._group_by_section(elements)
        groups = [g for g in groups if len(g.page_content) >= _MIN_GROUP_CHARS]

        if len(groups) >= _MIN_GROUPS:
            return groups

        combined = "\n\n".join(str(el) for el in elements if str(el).strip())
        return [Document(page_content=combined, metadata={"source": self.url})]

    def _group_by_section(self, elements: list) -> list[Document]:
        groups: list[Document] = []
        buffer: list[str] = []

        def flush() -> None:
            text = "\n\n".join(p for p in buffer if p.strip())
            if text:
                groups.append(Document(page_content=text, metadata={"source": self.url}))
            buffer.clear()

        for el in elements:
            if el.category in _HEADER_CATEGORIES:
                flush()
            buffer.append(str(el))

        flush()
        return groups


assert isinstance(WebLoader(""), Loader)
