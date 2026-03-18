"""
Qdrant vector store backend.
Requires: pip install qdrant-client
"""

from raglib.stores.vector.base import BaseVectorStore
from raglib.models.retrieval import SemanticChunk, CommunityInfo, RetrievalResult


class QdrantStore(BaseVectorStore):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str = "",
        collection_name: str = "raglib_chunks",
        community_collection: str = "raglib_communities",
        vector_size: int = 768,
    ):
        self._url = url
        self._api_key = api_key
        self._collection_name = collection_name
        self._community_collection = community_collection
        self._vector_size = vector_size

    def _get_client(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if not hasattr(self, "_client"):
            self._client = QdrantClient(url=self._url, api_key=self._api_key or None)
            for cname in [self._collection_name, self._community_collection]:
                try:
                    self._client.get_collection(cname)
                except Exception:
                    self._client.create_collection(
                        collection_name=cname,
                        vectors_config=VectorParams(
                            size=self._vector_size, distance=Distance.COSINE
                        ),
                    )
        return self._client

    # ── Chunk Operations ────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        from qdrant_client.models import PointStruct

        client = self._get_client()
        points = [
            PointStruct(
                id=abs(hash(c.chunk_id)) % (2**63),
                vector=c.embedding,
                payload={
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
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
        if points:
            client.upsert(collection_name=self._collection_name, points=points)

    def query_chunks(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        qfilter = None
        if filters:
            must_clauses = []
            if "doc_id" in filters:
                must_clauses.append(
                    FieldCondition(key="doc_id", match=MatchValue(value=filters["doc_id"]))
                )
            if "page_num" in filters:
                must_clauses.append(
                    FieldCondition(
                        key="page_num", match=MatchValue(value=filters["page_num"])
                    )
                )
            if must_clauses:
                qfilter = Filter(must=must_clauses)

        results = client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [
            RetrievalResult(
                content=r.payload.get("text", ""),
                score=r.score,
                source="vector",
                metadata=r.payload,
            )
            for r in results
        ]

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        if not chunk_ids:
            return []
        client = self._get_client()
        results = []
        for cid in chunk_ids:
            int_id = abs(hash(cid)) % (2**63)
            pts = client.retrieve(
                collection_name=self._collection_name,
                ids=[int_id],
                with_payload=True,
            )
            for p in pts:
                results.append(
                    RetrievalResult(
                        content=p.payload.get("text", ""),
                        score=0.8,
                        source="vector_bridge",
                        metadata=p.payload,
                    )
                )
        return results

    def delete_by_doc_id(self, doc_id: str):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    # ── Community Operations ────────────────────────────────────────────

    def upsert_community(self, community: CommunityInfo):
        if not community.embedding:
            return
        from qdrant_client.models import PointStruct

        client = self._get_client()
        point = PointStruct(
            id=abs(hash(community.community_id)) % (2**63),
            vector=community.embedding,
            payload={
                "community_id": community.community_id,
                "text": f"{community.title}: {community.summary}",
                "level": community.level,
                "rank": community.rank,
                "entity_count": len(community.entity_names),
            },
        )
        client.upsert(collection_name=self._community_collection, points=[point])

    def query_communities(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        client = self._get_client()
        try:
            coll = client.get_collection(self._community_collection)
            if coll.points_count == 0:
                return []
        except Exception:
            return []
        results = client.search(
            collection_name=self._community_collection,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievalResult(
                content=r.payload.get("text", ""),
                score=r.score,
                source="community",
                metadata=r.payload,
            )
            for r in results
        ]

    def count(self) -> dict:
        client = self._get_client()
        try:
            chunks = client.get_collection(self._collection_name).points_count
        except Exception:
            chunks = 0
        try:
            communities = client.get_collection(
                self._community_collection
            ).points_count
        except Exception:
            communities = 0
        return {"chunks": chunks, "communities": communities}
