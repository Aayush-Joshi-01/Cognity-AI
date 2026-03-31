"""
ChromaDB vector store with dual collections:
  1. Chunk collection — semantic chunks with page/entity metadata
  2. Community collection — community summaries for global search

Direct migration of hybrid_rag/vector_manager.py VectorManager.
"""

import chromadb
from chromadb.config import Settings

from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.models.retrieval import SemanticChunk, CommunityInfo, RetrievalResult
from cognity_ai.config.providers import ChromaConfig


class ChromaStore(BaseVectorStore):
    def __init__(self, config: ChromaConfig):
        self.client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.chunks = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.communities = self.client.get_or_create_collection(
            name=config.community_collection,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Chunk Operations ────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        if not chunks:
            return
        self.chunks.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{
                "doc_id": c.doc_id,
                "index": c.index,
                "page_num": c.page_info.page_num if c.page_info else 0,
                "section": c.page_info.section if c.page_info else "",
                "heading": c.page_info.heading if c.page_info else "",
                "entity_names": "|".join(c.entity_names),  # searchable string
                "sentence_count": c.sentence_count,
                "token_estimate": c.token_estimate,
            } for c in chunks],
        )

    def delete_by_doc(self, doc_id: str):
        """Delete all chunks for a document (original method name)."""
        try:
            self.chunks.delete(where={"doc_id": doc_id})
        except Exception:
            results = self.chunks.get(where={"doc_id": doc_id})
            if results["ids"]:
                self.chunks.delete(ids=results["ids"])

    def delete_by_doc_id(self, doc_id: str):
        """Alias of delete_by_doc for BaseVectorStore compliance."""
        self.delete_by_doc(doc_id)

    def query_chunks(self, embedding: list[float], top_k: int = 10,
                     filters: dict | None = None,
                     filter_doc_ids: list[str] | None = None,
                     filter_page: int | None = None) -> list[RetrievalResult]:
        """
        Query chunks by embedding similarity.

        Accepts the BaseVectorStore `filters` dict (keys: doc_id, page_num)
        as well as the original keyword arguments for backwards compat.
        """
        # Normalise filters: merge the `filters` dict with legacy kwargs
        if filters:
            if "doc_id" in filters and filter_doc_ids is None:
                filter_doc_ids = [filters["doc_id"]]
            if "doc_ids" in filters and filter_doc_ids is None:
                filter_doc_ids = filters["doc_ids"]
            if "page_num" in filters and filter_page is None:
                filter_page = filters["page_num"]

        where = {}
        where_clauses = []
        if filter_doc_ids:
            where_clauses.append({"doc_id": {"$in": filter_doc_ids}})
        if filter_page is not None:
            where_clauses.append({"page_num": filter_page})

        if len(where_clauses) > 1:
            where = {"$and": where_clauses}
        elif where_clauses:
            where = where_clauses[0]

        results = self.chunks.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        retrieval = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            retrieval.append(RetrievalResult(
                content=doc,
                score=similarity,
                source="vector",
                metadata=meta,
            ))
        return retrieval

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        """Direct retrieval by chunk IDs (used for graph→vector bridge)."""
        if not chunk_ids:
            return []
        results = self.chunks.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        retrieval = []
        for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            retrieval.append(RetrievalResult(
                content=doc,
                score=0.8,  # direct lookup gets a base score
                source="vector_bridge",
                metadata=meta,
            ))
        return retrieval

    # ── Community Operations ────────────────────────────────────────────

    def upsert_community(self, community: CommunityInfo):
        if not community.embedding:
            return
        self.communities.upsert(
            ids=[community.community_id],
            embeddings=[community.embedding],
            documents=[f"{community.title}: {community.summary}"],
            metadatas=[{
                "level": community.level,
                "rank": community.rank,
                "entity_count": len(community.entity_names),
            }],
        )

    def query_communities(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        if self.communities.count() == 0:
            return []
        results = self.communities.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.communities.count()),
            include=["documents", "metadatas", "distances"],
        )
        retrieval = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieval.append(RetrievalResult(
                content=doc,
                score=1.0 - dist,
                source="community",
                metadata=meta,
            ))
        return retrieval

    def count(self) -> dict:
        return {
            "chunks": self.chunks.count(),
            "communities": self.communities.count(),
        }
