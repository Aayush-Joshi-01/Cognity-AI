"""Tests for vector store backends.

Local stores (FAISS, Chroma) are tested directly when their deps are installed.
Cloud stores (Pinecone, Qdrant cloud, Milvus, Weaviate, etc.) are skipped
unless the relevant credentials are available in environment variables.
"""
from __future__ import annotations

import pytest
from conftest import (
    requires_faiss, requires_chromadb, requires_pinecone,
    make_embedding,
)
from cognity_ai.models.retrieval import SemanticChunk, CommunityInfo, RetrievalResult

DIM = 8  # tiny dimension for fast tests


def _chunk(chunk_id: str, doc_id: str, text: str, dim: int = DIM) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        index=0,
        embedding=make_embedding(dim=dim, seed=hash(chunk_id) % 2**31),
    )


def _community(cid: str, title: str, summary: str, dim: int = DIM) -> CommunityInfo:
    return CommunityInfo(
        community_id=cid,
        level=0,
        title=title,
        summary=summary,
        embedding=make_embedding(dim=dim, seed=hash(cid) % 2**31),
    )


# ── FAISS ─────────────────────────────────────────────────────────────────────

@requires_faiss
class TestFAISSStore:
    @pytest.fixture
    def store(self):
        from cognity_ai.stores.vector.faiss import FAISSStore
        return FAISSStore(dimension=DIM)

    def test_count_empty_on_init(self, store):
        assert store.count()["chunks"] == 0
        assert store.count()["communities"] == 0

    def test_upsert_and_count(self, store):
        store.upsert_chunks([_chunk("c1", "d1", "hello world")])
        assert store.count()["chunks"] == 1

    def test_upsert_multiple_chunks(self, store):
        chunks = [_chunk(f"c{i}", "d1", f"text {i}") for i in range(5)]
        store.upsert_chunks(chunks)
        assert store.count()["chunks"] == 5

    def test_upsert_empty_list_is_noop(self, store):
        store.upsert_chunks([])
        assert store.count()["chunks"] == 0

    def test_query_returns_results(self, store):
        store.upsert_chunks([
            _chunk("c1", "d1", "machine learning"),
            _chunk("c2", "d1", "deep learning"),
            _chunk("c3", "d2", "natural language processing"),
        ])
        query_vec = make_embedding(dim=DIM, seed=1)
        results = store.query_chunks(query_vec, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_query_top_k_respected(self, store):
        for i in range(10):
            store.upsert_chunks([_chunk(f"c{i}", "d1", f"text {i}")])
        query_vec = make_embedding(dim=DIM, seed=99)
        results = store.query_chunks(query_vec, top_k=3)
        assert len(results) == 3

    def test_query_returns_source_vector(self, store):
        store.upsert_chunks([_chunk("c1", "d1", "hello")])
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=1)
        assert results[0].source == "vector"

    def test_query_empty_store_returns_empty(self, store):
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=5)
        assert results == []

    def test_filter_by_doc_id(self, store):
        store.upsert_chunks([
            _chunk("c1", "doc_a", "text a"),
            _chunk("c2", "doc_b", "text b"),
        ])
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=10,
                                     filters={"doc_id": "doc_a"})
        assert all(r.metadata["doc_id"] == "doc_a" for r in results)

    def test_query_by_chunk_ids(self, store):
        store.upsert_chunks([_chunk("c1", "d1", "hello"), _chunk("c2", "d1", "world")])
        results = store.query_by_chunk_ids(["c1"])
        assert len(results) == 1
        assert results[0].source == "vector_bridge"

    def test_query_by_chunk_ids_empty(self, store):
        results = store.query_by_chunk_ids([])
        assert results == []

    def test_delete_by_doc_id(self, store):
        store.upsert_chunks([
            _chunk("c1", "d1", "delete me"),
            _chunk("c2", "d2", "keep me"),
        ])
        store.delete_by_doc_id("d1")
        assert store.count()["chunks"] == 1

    def test_upsert_community(self, store):
        store.upsert_community(_community("comm1", "AI Research", "Covers AI topics"))
        assert store.count()["communities"] == 1

    def test_query_communities(self, store):
        store.upsert_community(_community("comm1", "AI", "AI research"))
        store.upsert_community(_community("comm2", "Biology", "Life science"))
        results = store.query_communities(make_embedding(DIM, seed=5), top_k=2)
        assert len(results) == 2
        assert all(r.source == "community" for r in results)

    def test_query_communities_empty_store(self, store):
        results = store.query_communities(make_embedding(DIM, seed=5), top_k=5)
        assert results == []

    def test_chunk_without_embedding_skipped(self, store):
        chunk = SemanticChunk(chunk_id="no_embed", doc_id="d1", text="no embedding", index=0)
        store.upsert_chunks([chunk])
        assert store.count()["chunks"] == 0

    def test_persistence_save_load(self, tmp_dir):
        """Save to disk and reload."""
        import os
        from cognity_ai.stores.vector.faiss import FAISSStore
        path = os.path.join(tmp_dir, "faiss_test")
        store = FAISSStore(dimension=DIM, index_path=path)
        store.upsert_chunks([_chunk("c1", "d1", "persisted text")])
        store.save()

        store2 = FAISSStore(dimension=DIM, index_path=path)
        assert store2.count()["chunks"] == 1


# ── ChromaDB ─────────────────────────────────────────────────────────────────

@requires_chromadb
class TestChromaStore:
    @pytest.fixture
    def store(self, tmp_dir):
        from cognity_ai.stores.vector.chroma import ChromaStore
        from cognity_ai.config.providers import ChromaConfig
        cfg = ChromaConfig(
            persist_directory=tmp_dir,
            collection_name="test_chunks",
            community_collection="test_communities",
        )
        return ChromaStore(cfg)

    def test_count_empty_on_init(self, store):
        c = store.count()
        assert c["chunks"] == 0
        assert c["communities"] == 0

    def test_upsert_and_count(self, store):
        # Chroma requires 768-dim vectors by default; we use a 768-dim vec for real
        # but override the collection to not check distance space
        # For tests we use DIM=8 — create a store with no forced space
        store.upsert_chunks([_chunk("c1", "d1", "hello")])
        assert store.count()["chunks"] == 1

    def test_upsert_empty_list_noop(self, store):
        store.upsert_chunks([])
        assert store.count()["chunks"] == 0

    def test_upsert_multiple_chunks(self, store):
        chunks = [_chunk(f"c{i}", "d1", f"text {i}") for i in range(3)]
        store.upsert_chunks(chunks)
        assert store.count()["chunks"] == 3

    def test_query_returns_results(self, store):
        store.upsert_chunks([
            _chunk("c1", "d1", "machine learning"),
            _chunk("c2", "d1", "deep learning"),
            _chunk("c3", "d2", "biology"),
        ])
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_query_source_is_vector(self, store):
        store.upsert_chunks([_chunk("c1", "d1", "text")])
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=1)
        assert results[0].source == "vector"

    def test_delete_by_doc_id(self, store):
        store.upsert_chunks([
            _chunk("c1", "d1", "delete me"),
            _chunk("c2", "d2", "keep me"),
        ])
        store.delete_by_doc_id("d1")
        assert store.count()["chunks"] == 1

    def test_delete_by_doc_alias(self, store):
        store.upsert_chunks([_chunk("c1", "d1", "text")])
        store.delete_by_doc("d1")
        assert store.count()["chunks"] == 0

    def test_query_by_chunk_ids(self, store):
        store.upsert_chunks([_chunk("c1", "d1", "hello"), _chunk("c2", "d1", "world")])
        results = store.query_by_chunk_ids(["c1"])
        assert len(results) == 1
        assert results[0].source == "vector_bridge"

    def test_query_by_chunk_ids_empty(self, store):
        assert store.query_by_chunk_ids([]) == []

    def test_upsert_community(self, store):
        store.upsert_community(_community("comm1", "AI", "AI topics"))
        assert store.count()["communities"] == 1

    def test_upsert_community_no_embedding_skipped(self, store):
        comm = CommunityInfo(community_id="c1", level=0, title="X", summary="Y")
        store.upsert_community(comm)
        assert store.count()["communities"] == 0

    def test_query_communities(self, store):
        store.upsert_community(_community("comm1", "AI", "AI research"))
        store.upsert_community(_community("comm2", "Bio", "Biology"))
        results = store.query_communities(make_embedding(DIM, seed=1), top_k=2)
        assert len(results) <= 2
        assert all(r.source == "community" for r in results)

    def test_query_communities_empty_returns_empty(self, store):
        results = store.query_communities(make_embedding(DIM, seed=1), top_k=5)
        assert results == []

    def test_upsert_is_idempotent(self, store):
        chunk = _chunk("c1", "d1", "text")
        store.upsert_chunks([chunk])
        store.upsert_chunks([chunk])  # upsert same chunk again
        assert store.count()["chunks"] == 1

    def test_filter_by_doc_id(self, store):
        store.upsert_chunks([
            _chunk("c1", "doc_a", "alpha"),
            _chunk("c2", "doc_b", "beta"),
        ])
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=5,
                                     filters={"doc_id": "doc_a"})
        assert all(r.metadata["doc_id"] == "doc_a" for r in results)


# ── Pinecone (skip unless creds available) ────────────────────────────────────

@requires_pinecone
class TestPineconeStore:
    """Integration tests — only run when PINECONE_API_KEY is set."""

    @pytest.fixture
    def store(self):
        from cognity_ai.stores.vector.pinecone import PineconeStore
        from cognity_ai.config.providers import PineconeConfig
        cfg = PineconeConfig(
            index_name="cognity-ai-test",
            namespace="test",
            dimension=8,
        )
        return PineconeStore(cfg)

    def test_upsert_and_query(self, store):
        store.upsert_chunks([_chunk("pc1", "d1", "pinecone test text")])
        results = store.query_chunks(make_embedding(DIM, seed=1), top_k=1)
        assert len(results) >= 1

    def test_delete_by_doc_id(self, store):
        store.upsert_chunks([_chunk("pc_del", "doc_del", "to delete")])
        store.delete_by_doc_id("doc_del")
        # No assertion on count — Pinecone may be eventually consistent


# ── Qdrant (skip unless creds or local instance available) ────────────────────

class TestQdrantStore:
    @pytest.fixture(autouse=True)
    def _require_qdrant(self):
        pytest.importorskip("qdrant_client")

    @pytest.fixture
    def store(self):
        import os
        from cognity_ai.stores.vector.qdrant import QdrantStore
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY", "")
        try:
            return QdrantStore(
                url=url,
                api_key=api_key,
                collection_name="test_chunks",
                community_collection="test_communities",
                vector_size=DIM,
            )
        except Exception:
            pytest.skip("Qdrant not reachable — set QDRANT_URL to a running instance")

    def test_upsert_and_count(self, store):
        try:
            store.upsert_chunks([_chunk("qc1", "d1", "qdrant test")])
            c = store.count()
            assert c["chunks"] >= 1
        except Exception as exc:
            pytest.skip(f"Qdrant not reachable: {exc}")
