"""Abstract base class for graph store backends."""
from abc import ABC, abstractmethod
from raglib.models.knowledge import Entity, Relation
from raglib.models.retrieval import RetrievalResult, CommunityInfo


class BaseGraphStore(ABC):
    @abstractmethod
    def upsert_entity(self, entity: Entity): ...

    @abstractmethod
    def upsert_relation(self, relation: Relation): ...

    @abstractmethod
    def link_chunk_to_entities(self, chunk_id: str, doc_id: str, entity_names: list[str]): ...

    @abstractmethod
    def retrieve_subgraph(self, entity_names: list[str], hops: int = 2, limit: int = 20) -> list[RetrievalResult]: ...

    @abstractmethod
    def retrieve_entity_context(self, entity_name: str) -> list[RetrievalResult]: ...

    @abstractmethod
    def global_community_search(self, top_n: int = 5) -> list[RetrievalResult]: ...

    @abstractmethod
    def get_chunks_for_entities(self, entity_names: list[str]) -> list[str]: ...

    @abstractmethod
    def detect_communities(self) -> list[dict]: ...

    @abstractmethod
    def get_community_entities(self, community_id) -> list[dict]: ...

    @abstractmethod
    def store_community_summary(self, community: CommunityInfo): ...

    @abstractmethod
    def upsert_doc_meta(self, doc_id: str, content_hash: str, source_name: str, status: str, stats: dict | None): ...

    @abstractmethod
    def remove_doc_subgraph(self, doc_id: str): ...

    @abstractmethod
    def confirm_source(self, doc_id: str): ...

    @abstractmethod
    def deprecate_source(self, doc_id: str): ...

    @abstractmethod
    def get_doc_status(self, doc_id: str) -> str | None: ...

    @abstractmethod
    def prune_low_confidence(self, threshold: float = 0.5) -> int: ...

    @abstractmethod
    def health_report(self) -> dict: ...
