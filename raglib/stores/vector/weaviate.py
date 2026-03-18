"""Weaviate vector store backend. Install: pip install weaviate-client>=4.0"""
from __future__ import annotations
from raglib.stores.vector.base import BaseVectorStore
from raglib.models.retrieval import RetrievalResult, SemanticChunk, CommunityInfo


class WeaviateStore(BaseVectorStore):
    def __init__(self, weaviate_config):
        try:
            import weaviate
        except ImportError:
            raise ImportError("weaviate-client not installed. Run: pip install weaviate-client>=4.0")

        import weaviate
        url = getattr(weaviate_config, "url", "http://localhost:8080")
        api_key = getattr(weaviate_config, "api_key", "")

        if api_key:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key),
            )
        else:
            self._client = weaviate.connect_to_local(
                host=url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(url.split(":")[-1]) if ":" in url else 8080,
            )

        self._class_name = getattr(weaviate_config, "class_name", "RaglibChunk")
        self._community_class = getattr(weaviate_config, "community_class", "RaglibCommunity")
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            if not self._client.collections.exists(self._class_name):
                self._client.collections.create(
                    name=self._class_name,
                    properties=[
                        {"name": "doc_id", "data_type": ["text"]},
                        {"name": "chunk_id", "data_type": ["text"]},
                        {"name": "text", "data_type": ["text"]},
                        {"name": "page_num", "data_type": ["int"]},
                        {"name": "entity_names", "data_type": ["text[]"]},
                        {"name": "is_parent", "data_type": ["boolean"]},
                        {"name": "parent_chunk_id", "data_type": ["text"]},
                    ],
                )
            if not self._client.collections.exists(self._community_class):
                self._client.collections.create(
                    name=self._community_class,
                    properties=[
                        {"name": "community_id", "data_type": ["text"]},
                        {"name": "title", "data_type": ["text"]},
                        {"name": "summary", "data_type": ["text"]},
                        {"name": "rank", "data_type": ["number"]},
                    ],
                )
        except Exception:
            pass

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        col = self._client.collections.get(self._class_name)
        with col.batch.dynamic() as batch:
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                props = {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "page_num": chunk.page_info.page_num if chunk.page_info else 0,
                    "entity_names": chunk.entity_names,
                    "is_parent": chunk.is_parent,
                    "parent_chunk_id": chunk.parent_chunk_id or "",
                }
                batch.add_object(properties=props, vector=chunk.embedding, uuid=chunk.chunk_id)

    def query_chunks(self, embedding: list[float], top_k: int = 10,
                     filters: dict = None) -> list[RetrievalResult]:
        col = self._client.collections.get(self._class_name)
        try:
            results = col.query.near_vector(
                near_vector=embedding, limit=top_k,
                return_metadata=["distance"],
                return_properties=["doc_id", "chunk_id", "text", "page_num", "entity_names"],
            )
            return [
                RetrievalResult(
                    content=obj.properties.get("text", ""),
                    score=1.0 - (obj.metadata.distance or 0),
                    source="vector",
                    metadata={
                        "doc_id": obj.properties.get("doc_id", ""),
                        "chunk_id": obj.properties.get("chunk_id", ""),
                        "page_num": obj.properties.get("page_num", 0),
                    },
                )
                for obj in results.objects
            ]
        except Exception:
            return []

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        col = self._client.collections.get(self._class_name)
        results = []
        for cid in chunk_ids:
            try:
                obj = col.query.fetch_object_by_id(cid)
                if obj:
                    results.append(RetrievalResult(
                        content=obj.properties.get("text", ""),
                        score=1.0, source="vector_bridge",
                        metadata={"chunk_id": cid, "doc_id": obj.properties.get("doc_id", "")},
                    ))
            except Exception:
                pass
        return results

    def upsert_community(self, community: CommunityInfo):
        if community.embedding is None:
            return
        col = self._client.collections.get(self._community_class)
        props = {
            "community_id": community.community_id,
            "title": community.title,
            "summary": community.summary,
            "rank": community.rank,
        }
        try:
            col.data.insert(properties=props, vector=community.embedding,
                            uuid=community.community_id)
        except Exception:
            try:
                col.data.update(uuid=community.community_id, properties=props,
                                vector=community.embedding)
            except Exception:
                pass

    def query_communities(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        col = self._client.collections.get(self._community_class)
        try:
            results = col.query.near_vector(near_vector=embedding, limit=top_k,
                                            return_metadata=["distance"],
                                            return_properties=["community_id", "title", "summary", "rank"])
            return [
                RetrievalResult(
                    content=f"[{obj.properties.get('title', '')}] {obj.properties.get('summary', '')}",
                    score=1.0 - (obj.metadata.distance or 0),
                    source="community",
                    metadata={"community_id": obj.properties.get("community_id", "")},
                )
                for obj in results.objects
            ]
        except Exception:
            return []

    def delete_by_doc_id(self, doc_id: str):
        try:
            import weaviate.classes.query as wq
            col = self._client.collections.get(self._class_name)
            col.data.delete_many(where=wq.Filter.by_property("doc_id").equal(doc_id))
        except Exception:
            pass
