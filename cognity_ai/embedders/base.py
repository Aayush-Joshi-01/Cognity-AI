"""Base ABC for all embedder implementations."""
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Embed a batch of texts. Returns a list of embedding vectors."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string with retrieval_query task type."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors produced."""
        ...
