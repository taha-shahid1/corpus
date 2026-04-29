from typing import Protocol, runtime_checkable

from langchain_core.documents import Document


@runtime_checkable
class Loader(Protocol):
    def load(self) -> list[Document]: ...
