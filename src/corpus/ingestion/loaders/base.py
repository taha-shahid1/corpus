from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Loader(Protocol):
    source: str

    def load(self) -> list[Document]: ...
