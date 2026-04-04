"""Tests for embedder implementations.

AI-backed embedders (Gemini, OpenAI, etc.) are skipped when credentials are absent.
SentenceTransformer embedder is skipped when the package is not installed.
"""
from __future__ import annotations

import os
import pytest
from conftest import (
    requires_gemini, requires_openai, requires_anthropic,
    requires_cohere, requires_aws, requires_azure_openai,
    requires_sentence_transformers,
)


def _assert_valid_embedding(emb, expected_dim: int | None = None):
    """Assert a valid float embedding vector."""
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, float) for x in emb)
    if expected_dim is not None:
        assert len(emb) == expected_dim


# ── GeminiEmbedder ────────────────────────────────────────────────────────────

@requires_gemini
class TestGeminiEmbedder:
    @pytest.fixture
    def embedder(self):
        pytest.importorskip("google.genai")
        from cognity_ai.embedders.gemini import GeminiEmbedder
        return GeminiEmbedder()  # reads GOOGLE_API_KEY from env

    def test_embed_query_returns_list(self, embedder):
        emb = embedder.embed_query("What is machine learning?")
        _assert_valid_embedding(emb)

    def test_embed_query_dimension(self, embedder):
        emb = embedder.embed_query("test")
        assert len(emb) == embedder.dimensions

    def test_dimensions_property(self, embedder):
        assert embedder.dimensions == 768

    def test_embed_batch_single(self, embedder):
        results = embedder.embed_batch(["hello world"])
        assert len(results) == 1
        _assert_valid_embedding(results[0])

    def test_embed_batch_multiple(self, embedder):
        texts = ["text one", "text two", "text three"]
        results = embedder.embed_batch(texts)
        assert len(results) == len(texts)
        for emb in results:
            _assert_valid_embedding(emb)

    def test_embed_batch_same_text_same_vector(self, embedder):
        emb1 = embedder.embed_batch(["hello"])[0]
        emb2 = embedder.embed_batch(["hello"])[0]
        assert emb1 == pytest.approx(emb2, abs=1e-5)

    def test_different_texts_different_vectors(self, embedder):
        emb_a = embedder.embed_query("cat")
        emb_b = embedder.embed_query("quantum physics")
        assert emb_a != pytest.approx(emb_b, abs=1e-3)

    def test_embed_query_vs_embed_batch_same_length(self, embedder):
        q = embedder.embed_query("test query")
        b = embedder.embed_batch(["test query"])[0]
        assert len(q) == len(b)

    def test_custom_model_via_config(self):
        pytest.importorskip("google.genai")
        from cognity_ai.config.providers import GeminiConfig
        from cognity_ai.embedders.gemini import GeminiEmbedder
        cfg = GeminiConfig()
        embedder = GeminiEmbedder(cfg)
        emb = embedder.embed_query("test")
        _assert_valid_embedding(emb)


# ── OpenAI Embedder ───────────────────────────────────────────────────────────

@requires_openai
class TestOpenAIEmbedder:
    @pytest.fixture
    def embedder(self):
        pytest.importorskip("openai")
        from cognity_ai.embedders.openai import OpenAIEmbedder
        from cognity_ai.config.providers import OpenAIConfig
        return OpenAIEmbedder(OpenAIConfig())

    def test_embed_query_returns_list(self, embedder):
        emb = embedder.embed_query("hello world")
        _assert_valid_embedding(emb)

    def test_embed_batch(self, embedder):
        results = embedder.embed_batch(["alpha", "beta"])
        assert len(results) == 2
        for r in results:
            _assert_valid_embedding(r)

    def test_dimensions(self, embedder):
        emb = embedder.embed_query("test")
        assert len(emb) == embedder.dimensions


# ── Cohere Embedder ───────────────────────────────────────────────────────────

@requires_cohere
class TestCohereEmbedder:
    @pytest.fixture
    def embedder(self):
        pytest.importorskip("cohere")
        from cognity_ai.embedders.cohere import CohereEmbedder
        from cognity_ai.config.providers import CohereConfig
        return CohereEmbedder(CohereConfig())

    def test_embed_query(self, embedder):
        emb = embedder.embed_query("test text")
        _assert_valid_embedding(emb)

    def test_embed_batch(self, embedder):
        results = embedder.embed_batch(["one", "two"])
        assert len(results) == 2


# ── SentenceTransformer Embedder ──────────────────────────────────────────────

@requires_sentence_transformers
class TestSentenceTransformerEmbedder:
    @pytest.fixture
    def embedder(self):
        from cognity_ai.embedders.sentence_transformers import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()

    def test_embed_query_returns_list(self, embedder):
        emb = embedder.embed_query("What is AI?")
        _assert_valid_embedding(emb)

    def test_dimensions_positive(self, embedder):
        assert embedder.dimensions > 0

    def test_embed_batch(self, embedder):
        texts = ["hello", "world"]
        results = embedder.embed_batch(texts)
        assert len(results) == 2
        for r in results:
            _assert_valid_embedding(r)

    def test_embed_batch_empty(self, embedder):
        results = embedder.embed_batch([])
        assert results == []

    def test_different_texts_different_vectors(self, embedder):
        emb_a = embedder.embed_query("cat sitting on a mat")
        emb_b = embedder.embed_query("astrophysics and black holes")
        assert emb_a != pytest.approx(emb_b, abs=1e-3)


# ── AWS Bedrock Embedder ──────────────────────────────────────────────────────

@requires_aws
class TestBedrockEmbedder:
    @pytest.fixture
    def embedder(self):
        pytest.importorskip("boto3")
        from cognity_ai.embedders.bedrock import BedrockEmbedder
        from cognity_ai.config.providers import BedrockConfig
        return BedrockEmbedder(BedrockConfig())

    def test_embed_query(self, embedder):
        emb = embedder.embed_query("bedrock test")
        _assert_valid_embedding(emb)

    def test_embed_batch(self, embedder):
        results = embedder.embed_batch(["text a", "text b"])
        assert len(results) == 2


# ── Azure OpenAI Embedder ─────────────────────────────────────────────────────

@requires_azure_openai
class TestAzureOpenAIEmbedder:
    @pytest.fixture
    def embedder(self):
        pytest.importorskip("openai")
        from cognity_ai.embedders.azure_openai import AzureOpenAIEmbedder
        from cognity_ai.config.providers import AzureOpenAIConfig
        return AzureOpenAIEmbedder(AzureOpenAIConfig())

    def test_embed_query(self, embedder):
        emb = embedder.embed_query("azure test")
        _assert_valid_embedding(emb)
