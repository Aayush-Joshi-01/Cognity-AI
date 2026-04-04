"""Tests for ComponentFactory (factory.py).

Uses MinimalLibraryConfig (faiss + networkx + fixed chunker) so no
API keys are required for the structural/routing tests.
Embedder and generator construction is patched so tests run offline.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch
from conftest import requires_faiss, requires_networkx, HAS_CHROMADB
from cognity_ai.config.base import LibraryConfig, MinimalLibraryConfig


# ── Shared mock builders ──────────────────────────────────────────────────────

def _mock_embedder():
    """Minimal embedder stub that returns 8-dim zero vectors."""
    m = MagicMock()
    m.embed_query.return_value = [0.0] * 8
    m.embed_batch.return_value = [[0.0] * 8]
    m.dimensions = 8
    return m


def _mock_generator():
    """Minimal generator stub."""
    m = MagicMock()
    m.generate.return_value = "mock answer"
    m.generate_rag.return_value = "mock rag answer"
    return m


def _patch_ai(func):
    """Decorator: patch embedder and generator construction in factory."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None):
            return func(*args, **kwargs)

    return wrapper


# ── build_components (structural tests, no AI calls) ─────────────────────────

class TestBuildComponentsMinimal:
    """Build the component dict with local-only providers.

    AI construction (embedder, generator, OCR) is patched so no API keys
    or model downloads are needed for these structural checks.
    """

    @pytest.fixture
    def components(self, tmp_path):
        pytest.importorskip("faiss")
        pytest.importorskip("networkx")
        from cognity_ai.factory import build_components

        cfg = MinimalLibraryConfig()
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        cfg.chroma.persist_directory = str(tmp_path / "chroma")

        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None):
            return build_components(cfg)

    def test_components_dict_has_required_keys(self, components):
        expected = {
            "nlp_model", "extractor", "chunker", "embedder",
            "vector_store", "graph_store", "generator",
            "page_index", "hash_store", "ocr", "rag_method",
        }
        assert expected.issubset(set(components.keys()))

    def test_vector_store_is_faiss(self, components):
        from cognity_ai.stores.vector.faiss import FAISSStore
        assert isinstance(components["vector_store"], FAISSStore)

    def test_graph_store_is_networkx(self, components):
        from cognity_ai.stores.graph.networkx import NetworkXStore
        assert isinstance(components["graph_store"], NetworkXStore)

    def test_chunker_is_fixed(self, components):
        from cognity_ai.chunkers.fixed import FixedChunker
        assert isinstance(components["chunker"], FixedChunker)

    def test_hash_store_present(self, components):
        from cognity_ai.utils.hash import HashStore
        assert isinstance(components["hash_store"], HashStore)

    def test_rag_method_is_vector_only(self, components):
        assert components["rag_method"] == "vector_only"

    def test_nlp_model_is_none_for_llm_only_extraction(self, components):
        # MinimalLibraryConfig uses extraction=llm_only → no spaCy loaded
        assert components["nlp_model"] is None


# ── Build with chroma vector store ───────────────────────────────────────────

class TestBuildComponentsChroma:
    @pytest.fixture
    def components(self, tmp_path):
        pytest.importorskip("chromadb")
        pytest.importorskip("networkx")
        from cognity_ai.factory import build_components
        cfg = MinimalLibraryConfig(vector_store="chroma")
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        cfg.chroma.persist_directory = str(tmp_path / "chroma")
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None):
            return build_components(cfg)

    def test_vector_store_is_chroma(self, components):
        from cognity_ai.stores.vector.chroma import ChromaStore
        assert isinstance(components["vector_store"], ChromaStore)


# ── Graph store fallback behaviour ────────────────────────────────────────────

class TestGraphStoreFallback:
    def test_graph_store_none_when_set_to_none(self, tmp_path):
        pytest.importorskip("faiss")
        from cognity_ai.factory import build_components
        cfg = MinimalLibraryConfig(graph_store="none")
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None):
            comps = build_components(cfg)
        assert comps["graph_store"] is None

    def test_hybrid_graph_falls_back_to_naive_without_graph(self, tmp_path):
        pytest.importorskip("faiss")
        import warnings
        from cognity_ai.factory import build_components
        cfg = MinimalLibraryConfig(rag_method="hybrid_graph", graph_store="none")
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None), \
             warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            comps = build_components(cfg)
        assert comps["rag_method"] == "naive"
        assert any("hybrid_graph" in str(warning.message) for warning in w)


# ── Anthropic embedder auto-switch ────────────────────────────────────────────

class TestAnthropicEmbedderSwitch:
    def test_anthropic_embedder_warning_is_emitted(self, tmp_path):
        """build_components warns when embedder='anthropic' and switches to sentence_transformers."""
        pytest.importorskip("faiss")
        import warnings
        from cognity_ai.factory import _build_embedder
        from cognity_ai.config.base import MinimalLibraryConfig

        cfg = MinimalLibraryConfig(embedder="anthropic")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from cognity_ai.factory import build_components
            with patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
                 patch("cognity_ai.factory._build_ocr", return_value=None):
                cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
                cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
                try:
                    build_components(cfg)
                except Exception:
                    pass  # expected if sentence_transformers not installed
        assert any("Anthropic" in str(warning.message) for warning in w)


# ── build_retriever ───────────────────────────────────────────────────────────

class TestBuildRetriever:
    @pytest.fixture
    def components(self, tmp_path):
        pytest.importorskip("faiss")
        pytest.importorskip("networkx")
        from cognity_ai.factory import build_components
        cfg = MinimalLibraryConfig()
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None):
            return build_components(cfg), cfg

    def test_naive_retriever(self, components):
        from cognity_ai.factory import build_retriever
        from cognity_ai.retrievers.naive import NaiveRetriever
        comps, cfg = components
        retriever = build_retriever("naive", comps, cfg)
        assert isinstance(retriever, NaiveRetriever)

    def test_vector_only_retriever(self, components):
        from cognity_ai.factory import build_retriever
        from cognity_ai.retrievers.vector_only import VectorOnlyRetriever
        comps, cfg = components
        retriever = build_retriever("vector_only", comps, cfg)
        assert isinstance(retriever, VectorOnlyRetriever)

    def test_hybrid_graph_retriever(self, components):
        from cognity_ai.factory import build_retriever
        from cognity_ai.retrievers.hybrid_graph import HybridGraphRetriever
        comps, cfg = components
        retriever = build_retriever("hybrid_graph", comps, cfg)
        assert isinstance(retriever, HybridGraphRetriever)

    def test_invalid_rag_method_raises(self, components):
        from cognity_ai.factory import build_retriever
        comps, cfg = components
        with pytest.raises(ValueError):
            build_retriever("nonexistent_method", comps, cfg)

    def test_adaptive_retriever(self, components):
        from cognity_ai.factory import build_retriever
        from cognity_ai.retrievers.adaptive import AdaptiveRetriever
        comps, cfg = components
        retriever = build_retriever("adaptive", comps, cfg)
        assert isinstance(retriever, AdaptiveRetriever)


# ── Chunker selection ────────────────────────────────────────────────────────

class TestChunkerSelection:
    @pytest.fixture(autouse=True)
    def _require_local_deps(self):
        pytest.importorskip("faiss")
        pytest.importorskip("networkx")

    def _build(self, tmp_path, chunker: str):
        from cognity_ai.factory import build_components
        cfg = MinimalLibraryConfig(chunker=chunker)
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None):
            return build_components(cfg)

    def test_fixed_chunker(self, tmp_path):
        from cognity_ai.chunkers.fixed import FixedChunker
        comps = self._build(tmp_path, "fixed")
        assert isinstance(comps["chunker"], FixedChunker)

    def test_recursive_chunker(self, tmp_path):
        from cognity_ai.chunkers.recursive import RecursiveChunker
        comps = self._build(tmp_path, "recursive")
        assert isinstance(comps["chunker"], RecursiveChunker)

    def test_invalid_chunker_raises(self, tmp_path):
        from cognity_ai.factory import build_components
        cfg = MinimalLibraryConfig(chunker="nonexistent")
        cfg.ingestion.hash_store_path = str(tmp_path / "hashes.json")
        cfg.ingestion.page_index_path = str(tmp_path / "page_index.json")
        with patch("cognity_ai.factory._build_embedder", return_value=_mock_embedder()), \
             patch("cognity_ai.factory._build_generator", return_value=_mock_generator()), \
             patch("cognity_ai.factory._build_ocr", return_value=None), \
             pytest.raises(ValueError):
            build_components(cfg)
