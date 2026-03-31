"""Azure AI Search vector store. Install: pip install azure-search-documents>=11.6"""
from __future__ import annotations
from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.models.retrieval import RetrievalResult, SemanticChunk, CommunityInfo


class AzureSearchStore(BaseVectorStore):
    def __init__(self, azure_search_config):
        try:
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            raise ImportError(
                "azure-search-documents not installed. Run: pip install azure-search-documents>=11.6"
            )

        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential

        endpoint = getattr(azure_search_config, "endpoint", "")
        api_key = getattr(azure_search_config, "api_key", "")
        self._index = getattr(azure_search_config, "index_name", "raglib-chunks")
        self._comm_index = getattr(azure_search_config, "community_index", "raglib-communities")
        self._dim = getattr(azure_search_config, "dimension", 1536)

        credential = AzureKeyCredential(api_key)
        self._index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        self._search_client = SearchClient(endpoint=endpoint, index_name=self._index,
                                           credential=credential)
        self._comm_client = SearchClient(endpoint=endpoint, index_name=self._comm_index,
                                         credential=credential)
        self._ensure_indexes(endpoint, credential)

    def _ensure_indexes(self, endpoint, credential):
        from azure.search.documents.indexes.models import (
            SearchIndex, SearchField, SearchFieldDataType,
            VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
            SimpleField, SearchableField,
        )
        for idx_name in [self._index, self._comm_index]:
            try:
                self._index_client.get_index(idx_name)
            except Exception:
                fields = [
                    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                    SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
                    SearchableField(name="text", type=SearchFieldDataType.String),
                    SimpleField(name="page_num", type=SearchFieldDataType.Int32),
                    SearchField(
                        name="embedding",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=self._dim,
                        vector_search_profile_name="hnsw-profile",
                    ),
                ]
                vector_search = VectorSearch(
                    algorithms=[HnswAlgorithmConfiguration(name="hnsw-alg")],
                    profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-alg")],
                )
                idx = SearchIndex(name=idx_name, fields=fields, vector_search=vector_search)
                try:
                    self._index_client.create_index(idx)
                except Exception:
                    pass

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        documents = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            documents.append({
                "id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "page_num": chunk.page_info.page_num if chunk.page_info else 0,
                "embedding": chunk.embedding,
            })
        if documents:
            self._search_client.upload_documents(documents=documents)

    def query_chunks(self, embedding: list[float], top_k: int = 10,
                     filters: dict = None) -> list[RetrievalResult]:
        from azure.search.documents.models import VectorizedQuery
        try:
            vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k,
                                           fields="embedding")
            results = self._search_client.search(
                search_text=None, vector_queries=[vector_query],
                select=["id", "doc_id", "text", "page_num"], top=top_k,
            )
            return [
                RetrievalResult(
                    content=r["text"],
                    score=r.get("@search.score", 0.5),
                    source="vector",
                    metadata={"chunk_id": r["id"], "doc_id": r["doc_id"],
                              "page_num": r.get("page_num", 0)},
                )
                for r in results
            ]
        except Exception:
            return []

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        results = []
        for cid in chunk_ids:
            try:
                doc = self._search_client.get_document(key=cid)
                results.append(RetrievalResult(
                    content=doc.get("text", ""), score=1.0, source="vector_bridge",
                    metadata={"chunk_id": cid, "doc_id": doc.get("doc_id", "")},
                ))
            except Exception:
                pass
        return results

    def upsert_community(self, community: CommunityInfo):
        if community.embedding is None:
            return
        self._comm_client.upload_documents(documents=[{
            "id": community.community_id,
            "doc_id": "community",
            "text": f"[{community.title}] {community.summary}",
            "page_num": 0,
            "embedding": community.embedding,
        }])

    def query_communities(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        from azure.search.documents.models import VectorizedQuery
        try:
            vq = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="embedding")
            results = self._comm_client.search(
                search_text=None, vector_queries=[vq], select=["id", "text"], top=top_k,
            )
            return [
                RetrievalResult(
                    content=r["text"], score=r.get("@search.score", 0.5),
                    source="community", metadata={"community_id": r["id"]},
                )
                for r in results
            ]
        except Exception:
            return []

    def delete_by_doc_id(self, doc_id: str):
        try:
            results = list(self._search_client.search(
                search_text="*", filter=f"doc_id eq '{doc_id}'", select=["id"]
            ))
            if results:
                self._search_client.delete_documents(documents=[{"id": r["id"]} for r in results])
        except Exception:
            pass
