"""
Pinecone vector store backend.
Requires: pip install pinecone-client
"""

from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.models.retrieval import SemanticChunk, CommunityInfo, RetrievalResult


class PineconeStore(BaseVectorStore):
    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "chunks",
        community_namespace: str = "communities",
        dimension: int = 1536,
    ):
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._community_namespace = community_namespace
        self._dimension = dimension

    def _get_index(self):
        from pinecone import Pinecone

        if not hasattr(self, "_index"):
            pc = Pinecone(api_key=self._api_key)
            self._index = pc.Index(self._index_name)
        return self._index

    # ── Chunk Operations ────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        index = self._get_index()
        vectors = [
            (
                c.chunk_id,
                c.embedding,
                {
                    "doc_id": c.doc_id,
                    "text": c.text[:8000],  # Pinecone metadata limit
                    "index": c.index,
                    "page_num": c.page_info.page_num if c.page_info else 0,
                    "section": c.page_info.section if c.page_info else "",
                    "heading": c.page_info.heading if c.page_info else "",
                    "entity_names": "|".join(c.entity_names),
                    "sentence_count": c.sentence_count,
                    "token_estimate": c.token_estimate,
                },
            )
            for c in chunks
            if c.embedding
        ]
        # Batch upsert 100 at a time
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i : i + 100], namespace=self._namespace)

    def query_chunks(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        index = self._get_index()
        pinecone_filter = {}
        if filters:
            if "doc_id" in filters:
                pinecone_filter["doc_id"] = {"$eq": filters["doc_id"]}
            if "page_num" in filters:
                pinecone_filter["page_num"] = {"$eq": filters["page_num"]}

        response = index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self._namespace,
            filter=pinecone_filter if pinecone_filter else None,
            include_metadata=True,
        )
        return [
            RetrievalResult(
                content=m.metadata.get("text", ""),
                score=m.score,
                source="vector",
                metadata=m.metadata,
            )
            for m in response.matches
        ]

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        if not chunk_ids:
            return []
        index = self._get_index()
        response = index.fetch(ids=chunk_ids, namespace=self._namespace)
        results = []
        for vid, vec in response.vectors.items():
            results.append(
                RetrievalResult(
                    content=vec.metadata.get("text", ""),
                    score=0.8,
                    source="vector_bridge",
                    metadata=vec.metadata,
                )
            )
        return results

    def delete_by_doc_id(self, doc_id: str):
        index = self._get_index()
        # Pinecone supports delete by metadata filter
        index.delete(
            filter={"doc_id": {"$eq": doc_id}},
            namespace=self._namespace,
        )

    # ── Community Operations ────────────────────────────────────────────

    def upsert_community(self, community: CommunityInfo):
        if not community.embedding:
            return
        index = self._get_index()
        index.upsert(
            vectors=[
                (
                    community.community_id,
                    community.embedding,
                    {
                        "community_id": community.community_id,
                        "text": f"{community.title}: {community.summary}"[:8000],
                        "level": community.level,
                        "rank": community.rank,
                        "entity_count": len(community.entity_names),
                    },
                )
            ],
            namespace=self._community_namespace,
        )

    def query_communities(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        index = self._get_index()
        try:
            stats = index.describe_index_stats()
            ns_stats = stats.namespaces.get(self._community_namespace)
            if not ns_stats or ns_stats.vector_count == 0:
                return []
        except Exception:
            return []

        response = index.query(
            vector=embedding,
            top_k=top_k,
            namespace=self._community_namespace,
            include_metadata=True,
        )
        return [
            RetrievalResult(
                content=m.metadata.get("text", ""),
                score=m.score,
                source="community",
                metadata=m.metadata,
            )
            for m in response.matches
        ]

    def count(self) -> dict:
        index = self._get_index()
        try:
            stats = index.describe_index_stats()
            chunks = (stats.namespaces.get(self._namespace) or type("", (), {"vector_count": 0})()).vector_count
            communities = (stats.namespaces.get(self._community_namespace) or type("", (), {"vector_count": 0})()).vector_count
        except Exception:
            chunks, communities = 0, 0
        return {"chunks": chunks, "communities": communities}
