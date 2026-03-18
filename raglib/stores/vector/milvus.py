"""Milvus vector store backend. Install: pip install pymilvus>=2.3"""
from __future__ import annotations
from raglib.stores.vector.base import BaseVectorStore
from raglib.models.retrieval import RetrievalResult, SemanticChunk, CommunityInfo


class MilvusStore(BaseVectorStore):
    def __init__(self, milvus_config):
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("pymilvus not installed. Run: pip install pymilvus>=2.3")

        from pymilvus import MilvusClient, DataType
        self._DataType = DataType

        uri = getattr(milvus_config, "uri", "http://localhost:19530")
        token = getattr(milvus_config, "token", "")
        self._collection = getattr(milvus_config, "collection_name", "raglib_chunks")
        self._comm_collection = getattr(milvus_config, "community_collection", "raglib_communities")
        self._dim = getattr(milvus_config, "dimension", 768)

        self._client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
        self._ensure_collections()

    def _ensure_collections(self):
        from pymilvus import DataType
        for col_name in [self._collection, self._comm_collection]:
            if not self._client.has_collection(col_name):
                schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
                schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)
                schema.add_field("text", DataType.VARCHAR, max_length=65535)
                schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
                index_params = self._client.prepare_index_params()
                index_params.add_index("embedding", index_type="HNSW",
                                       metric_type="COSINE", params={"M": 16, "efConstruction": 200})
                self._client.create_collection(
                    collection_name=col_name, schema=schema, index_params=index_params
                )

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        data = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            data.append({
                "id": chunk.chunk_id,
                "embedding": chunk.embedding,
                "text": chunk.text[:65530],
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_info.page_num if chunk.page_info else 0,
                "entity_names": ",".join(chunk.entity_names),
                "is_parent": int(chunk.is_parent),
                "parent_chunk_id": chunk.parent_chunk_id or "",
            })
        if data:
            self._client.upsert(collection_name=self._collection, data=data)

    def query_chunks(self, embedding: list[float], top_k: int = 10,
                     filters: dict = None) -> list[RetrievalResult]:
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[embedding],
                limit=top_k,
                output_fields=["text", "doc_id", "page_num", "entity_names"],
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            )
            return [
                RetrievalResult(
                    content=hit.get("entity", {}).get("text", ""),
                    score=hit.get("distance", 0),
                    source="vector",
                    metadata={
                        "doc_id": hit.get("entity", {}).get("doc_id", ""),
                        "chunk_id": hit.get("id", ""),
                        "page_num": hit.get("entity", {}).get("page_num", 0),
                    },
                )
                for hit in (results[0] if results else [])
            ]
        except Exception:
            return []

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        try:
            ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
            results = self._client.query(
                collection_name=self._collection,
                filter=f"id in [{ids_str}]",
                output_fields=["text", "doc_id"],
            )
            return [
                RetrievalResult(
                    content=r.get("text", ""),
                    score=1.0, source="vector_bridge",
                    metadata={"chunk_id": r.get("id", ""), "doc_id": r.get("doc_id", "")},
                )
                for r in results
            ]
        except Exception:
            return []

    def upsert_community(self, community: CommunityInfo):
        if community.embedding is None:
            return
        summary_text = f"[{community.title}] {community.summary}"
        self._client.upsert(collection_name=self._comm_collection, data=[{
            "id": community.community_id,
            "embedding": community.embedding,
            "text": summary_text[:65530],
            "doc_id": "community",
            "rank": community.rank,
        }])

    def query_communities(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        try:
            results = self._client.search(
                collection_name=self._comm_collection,
                data=[embedding], limit=top_k,
                output_fields=["text", "rank"],
                search_params={"metric_type": "COSINE"},
            )
            return [
                RetrievalResult(
                    content=hit.get("entity", {}).get("text", ""),
                    score=hit.get("distance", 0),
                    source="community",
                    metadata={"community_id": hit.get("id", "")},
                )
                for hit in (results[0] if results else [])
            ]
        except Exception:
            return []

    def delete_by_doc_id(self, doc_id: str):
        try:
            self._client.delete(collection_name=self._collection,
                                filter=f'doc_id == "{doc_id}"')
        except Exception:
            pass
