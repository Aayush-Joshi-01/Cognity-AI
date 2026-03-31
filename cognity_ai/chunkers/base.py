"""Abstract base class for text chunkers."""
from abc import ABC, abstractmethod
from cognity_ai.models.retrieval import SemanticChunk, PageInfo


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, doc_id: str, pages: list[PageInfo] | None = None) -> list[SemanticChunk]: ...
