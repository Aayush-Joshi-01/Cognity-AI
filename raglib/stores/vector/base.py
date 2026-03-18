"""Abstract base class for vector store backends."""
from abc import ABC, abstractmethod
from raglib.models.retrieval import SemanticChunk, CommunityInfo, RetrievalResult


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[SemanticChunk]): ...

    @abstractmethod
    def query_chunks(self, embedding: list[float], top_k: int = 10, filters: dict | None = None) -> list[RetrievalResult]: ...

    @abstractmethod
    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]: ...

    @abstractmethod
    def upsert_community(self, community: CommunityInfo): ...

    @abstractmethod
    def query_communities(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]: ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str): ...

    @abstractmethod
    def count(self) -> dict: ...
