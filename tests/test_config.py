"""Tests for LibraryConfig and MinimalLibraryConfig defaults."""
from __future__ import annotations

import pytest
from cognity_ai.config.base import LibraryConfig, MinimalLibraryConfig
from cognity_ai.config.providers import (
    GeminiConfig, ChromaConfig, Neo4jConfig, OpenAIConfig,
    AnthropicConfig, BedrockConfig, CohereConfig, OllamaConfig,
    QdrantConfig, PineconeConfig, MilvusConfig, WeaviateConfig,
    PgVectorConfig, AzureSearchConfig, NLPConfig, GraphRAGConfig,
    IngestionConfig, VertexAIConfig, AzureOpenAIConfig,
)


# ── LibraryConfig defaults ────────────────────────────────────────────────────

class TestLibraryConfigDefaults:
    def test_instantiation(self):
        cfg = LibraryConfig()
        assert cfg is not None

    def test_default_rag_method(self):
        assert LibraryConfig().rag_method == "hybrid_graph"

    def test_default_chunker(self):
        assert LibraryConfig().chunker == "sentence"

    def test_default_embedder(self):
        assert LibraryConfig().embedder == "gemini"

    def test_default_vector_store(self):
        assert LibraryConfig().vector_store == "chroma"

    def test_default_graph_store(self):
        assert LibraryConfig().graph_store == "neo4j"

    def test_default_llm(self):
        assert LibraryConfig().llm == "gemini"

    def test_default_extraction(self):
        assert LibraryConfig().extraction == "hybrid"

    def test_provider_configs_are_dataclasses(self):
        cfg = LibraryConfig()
        assert isinstance(cfg.gemini, GeminiConfig)
        assert isinstance(cfg.chroma, ChromaConfig)
        assert isinstance(cfg.neo4j, Neo4jConfig)
        assert isinstance(cfg.openai, OpenAIConfig)
        assert isinstance(cfg.anthropic, AnthropicConfig)
        assert isinstance(cfg.bedrock, BedrockConfig)
        assert isinstance(cfg.cohere, CohereConfig)
        assert isinstance(cfg.ollama, OllamaConfig)
        assert isinstance(cfg.qdrant, QdrantConfig)
        assert isinstance(cfg.pinecone, PineconeConfig)
        assert isinstance(cfg.milvus, MilvusConfig)
        assert isinstance(cfg.weaviate, WeaviateConfig)
        assert isinstance(cfg.pgvector, PgVectorConfig)
        assert isinstance(cfg.azure_search, AzureSearchConfig)
        assert isinstance(cfg.nlp, NLPConfig)
        assert isinstance(cfg.graphrag, GraphRAGConfig)
        assert isinstance(cfg.ingestion, IngestionConfig)

    def test_provider_configs_are_independent_instances(self):
        """Each LibraryConfig instance gets its own sub-config objects."""
        cfg1 = LibraryConfig()
        cfg2 = LibraryConfig()
        assert cfg1.gemini is not cfg2.gemini
        assert cfg1.chroma is not cfg2.chroma

    def test_override_rag_method(self):
        cfg = LibraryConfig(rag_method="naive")
        assert cfg.rag_method == "naive"

    def test_override_vector_store(self):
        cfg = LibraryConfig(vector_store="faiss")
        assert cfg.vector_store == "faiss"

    def test_override_graph_store_to_none(self):
        cfg = LibraryConfig(graph_store="none")
        assert cfg.graph_store == "none"


# ── GeminiConfig ─────────────────────────────────────────────────────────────

class TestGeminiConfig:
    def test_defaults(self):
        cfg = GeminiConfig(api_key="test-key")
        assert cfg.model == "gemini-2.0-flash"
        assert cfg.embedding_model == "models/text-embedding-004"
        assert cfg.temperature == 0.1
        assert cfg.extraction_temperature == 0.0
        assert cfg.batch_embed_limit == 100
        assert cfg.rpm_limit == 15
        assert cfg.timeout == 120
        assert cfg.use_vertexai is False

    def test_custom_values(self):
        cfg = GeminiConfig(api_key="k", model="gemini-1.5-pro", temperature=0.7)
        assert cfg.model == "gemini-1.5-pro"
        assert cfg.temperature == 0.7


# ── ChromaConfig ─────────────────────────────────────────────────────────────

class TestChromaConfig:
    def test_defaults(self):
        cfg = ChromaConfig()
        assert cfg.collection_name == "raglib_chunks"
        assert cfg.community_collection == "raglib_communities"

    def test_custom_persist_dir(self):
        cfg = ChromaConfig(persist_directory="/tmp/test_chroma")
        assert cfg.persist_directory == "/tmp/test_chroma"


# ── NLPConfig ────────────────────────────────────────────────────────────────

class TestNLPConfig:
    def test_defaults(self):
        cfg = NLPConfig()
        assert cfg.spacy_model == "en_core_web_trf"
        assert cfg.fallback_model == "en_core_web_sm"
        assert cfg.min_entity_freq == 1
        assert isinstance(cfg.dependency_relations, list)
        assert len(cfg.dependency_relations) > 0

    def test_dependency_relations_are_independent(self):
        cfg1 = NLPConfig()
        cfg2 = NLPConfig()
        cfg1.dependency_relations.append("foo")
        assert "foo" not in cfg2.dependency_relations


# ── GraphRAGConfig ───────────────────────────────────────────────────────────

class TestGraphRAGConfig:
    def test_defaults(self):
        cfg = GraphRAGConfig()
        assert cfg.leiden_resolution == 1.0
        assert cfg.max_community_levels == 3
        assert cfg.community_summary_max_tokens == 300
        assert cfg.local_search_top_k == 10
        assert cfg.global_search_top_communities == 5


# ── IngestionConfig ──────────────────────────────────────────────────────────

class TestIngestionConfig:
    def test_defaults(self):
        cfg = IngestionConfig()
        assert cfg.confidence_threshold == 0.5
        assert cfg.confirmed_boost == 1.5
        assert cfg.use_local_nlp_first is True
        assert cfg.gemini_extraction_mode == "augment"
        assert cfg.max_llm_chunks_per_doc == 50
        assert cfg.cache_embeddings is True


# ── MinimalLibraryConfig ─────────────────────────────────────────────────────

class TestMinimalLibraryConfig:
    def test_instantiation(self):
        cfg = MinimalLibraryConfig()
        assert cfg is not None

    def test_is_subclass_of_library_config(self):
        assert issubclass(MinimalLibraryConfig, LibraryConfig)

    def test_local_vector_store(self):
        assert MinimalLibraryConfig().vector_store == "faiss"

    def test_local_graph_store(self):
        assert MinimalLibraryConfig().graph_store == "networkx"

    def test_no_spacy_chunker(self):
        assert MinimalLibraryConfig().chunker == "fixed"

    def test_no_spacy_extraction(self):
        assert MinimalLibraryConfig().extraction == "llm_only"

    def test_local_rag_method(self):
        assert MinimalLibraryConfig().rag_method == "vector_only"

    def test_local_page_index(self):
        assert MinimalLibraryConfig().page_index == "regex"

    def test_still_has_all_provider_configs(self):
        """MinimalLibraryConfig inherits all provider sub-configs."""
        cfg = MinimalLibraryConfig()
        assert isinstance(cfg.gemini, GeminiConfig)
        assert isinstance(cfg.chroma, ChromaConfig)
        assert isinstance(cfg.ingestion, IngestionConfig)

    def test_can_override_fields(self):
        cfg = MinimalLibraryConfig(vector_store="chroma", graph_store="none")
        assert cfg.vector_store == "chroma"
        assert cfg.graph_store == "none"
