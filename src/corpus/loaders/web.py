import trafilatura
from langchain_core.documents import Document

from corpus.loaders.base import Loader


class WebLoader:
    def __init__(self, url: str) -> None:
        self.url = url

    def load(self) -> list[Document]:
        raw = trafilatura.fetch_url(self.url)
        if raw is None:
            raise ValueError(f"Failed to download {self.url!r}")

        text = trafilatura.extract(raw)
        if not text:
            raise ValueError(f"No extractable text at {self.url!r}")

        return [Document(page_content=text, metadata={"source": self.url})]


assert isinstance(WebLoader(""), Loader)
