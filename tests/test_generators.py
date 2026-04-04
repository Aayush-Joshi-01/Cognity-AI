"""Tests for LLM generator implementations.

All generators require external API credentials and are skipped when
the relevant environment variables are not set.
"""
from __future__ import annotations

import pytest
from conftest import (
    requires_gemini, requires_openai, requires_anthropic,
    requires_cohere, requires_aws, requires_azure_openai,
)
from cognity_ai.generators.base import BaseGenerator, GENERATION_PROMPT


# ── BaseGenerator helpers ─────────────────────────────────────────────────────

class TestBaseGeneratorHelpers:
    def test_build_rag_prompt_fills_all_placeholders(self):
        prompt = BaseGenerator.build_rag_prompt(
            question="What is X?",
            graph_ctx="Graph context here",
            community_ctx="Community context here",
            vector_ctx="Vector context here",
        )
        assert "What is X?" in prompt
        assert "Graph context here" in prompt
        assert "Community context here" in prompt
        assert "Vector context here" in prompt

    def test_build_rag_prompt_empty_contexts_use_fallback(self):
        prompt = BaseGenerator.build_rag_prompt("Q?", "", "", "")
        assert "No graph context." in prompt
        assert "No community context." in prompt
        assert "No document context." in prompt

    def test_generation_prompt_template_has_placeholders(self):
        assert "{graph_context}" in GENERATION_PROMPT
        assert "{community_context}" in GENERATION_PROMPT
        assert "{vector_context}" in GENERATION_PROMPT
        assert "{question}" in GENERATION_PROMPT


# ── GeminiGenerator ───────────────────────────────────────────────────────────

@requires_gemini
class TestGeminiGenerator:
    @pytest.fixture
    def gen(self):
        pytest.importorskip("google.genai")
        from cognity_ai.generators.gemini import GeminiGenerator
        return GeminiGenerator()

    def test_generate_returns_non_empty_string(self, gen):
        result = gen.generate("What is 2+2?", context="Mathematics basics.")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_generate_with_empty_question_uses_context_as_prompt(self, gen):
        result = gen.generate("", context="Say the word 'hello'.")
        assert isinstance(result, str)

    def test_generate_rag_returns_string(self, gen):
        result = gen.generate_rag(
            question="What does Alice do?",
            graph_ctx="Alice WORKS_AT Acme Corp.",
            community_ctx="",
            vector_ctx="Alice is a software engineer.",
        )
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_generate_with_structured_context(self, gen):
        result = gen.generate_with_structured_context(
            question="Who is Alice?",
            graph_context="Alice is a researcher.",
            vector_context="Alice works at MIT.",
        )
        assert isinstance(result, str)

    def test_generate_rag_uses_all_context_channels(self, gen):
        result = gen.generate_rag(
            question="Summarize.",
            graph_ctx="Node A connects to Node B.",
            community_ctx="Community: AI Research.",
            vector_ctx="Deep learning is powerful.",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_augment_extraction_returns_extraction_result(self, gen):
        from cognity_ai.models.knowledge import ExtractionResult
        existing = ExtractionResult()
        text = "Alice works at Acme Corp in New York. Bob is her manager."
        result = gen.augment_extraction(text, existing, source_id="test")
        assert hasattr(result, "entities")
        assert hasattr(result, "relations")

    def test_summarize_community_returns_title_and_summary(self, gen):
        result = gen.summarize_community(
            entity_names=["Alice", "Bob", "Acme Corp"],
            relation_descriptions=["Alice WORKS_AT Acme Corp", "Bob MANAGES Alice"],
        )
        assert isinstance(result, dict)
        assert "title" in result
        assert "summary" in result

    def test_custom_model_via_config(self):
        pytest.importorskip("google.genai")
        from cognity_ai.config.providers import GeminiConfig
        from cognity_ai.generators.gemini import GeminiGenerator
        cfg = GeminiConfig(model="gemini-2.0-flash")
        gen = GeminiGenerator(cfg)
        result = gen.generate("Say yes.", "Answer with yes.")
        assert isinstance(result, str)


# ── OpenAI Generator ──────────────────────────────────────────────────────────

@requires_openai
class TestOpenAIGenerator:
    @pytest.fixture
    def gen(self):
        pytest.importorskip("openai")
        from cognity_ai.generators.openai import OpenAIGenerator
        from cognity_ai.config.providers import OpenAIConfig
        return OpenAIGenerator(OpenAIConfig())

    def test_generate_returns_string(self, gen):
        result = gen.generate("What is 1+1?", context="Basic math.")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_generate_with_structured_context(self, gen):
        result = gen.generate_with_structured_context(
            question="Who is Alice?",
            graph_context="Alice WORKS_AT Acme.",
            vector_context="Alice is a scientist.",
        )
        assert isinstance(result, str)


# ── Anthropic Generator ───────────────────────────────────────────────────────

@requires_anthropic
class TestAnthropicGenerator:
    @pytest.fixture
    def gen(self):
        pytest.importorskip("anthropic")
        from cognity_ai.generators.anthropic import AnthropicGenerator
        from cognity_ai.config.providers import AnthropicConfig
        return AnthropicGenerator(AnthropicConfig())

    def test_generate_returns_string(self, gen):
        result = gen.generate("What is AI?", context="AI is a branch of computer science.")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_generate_with_structured_context(self, gen):
        result = gen.generate_with_structured_context(
            question="Summarize.",
            vector_context="Deep learning uses neural networks.",
        )
        assert isinstance(result, str)


# ── Cohere Generator ──────────────────────────────────────────────────────────

@requires_cohere
class TestCohereGenerator:
    @pytest.fixture
    def gen(self):
        pytest.importorskip("cohere")
        from cognity_ai.generators.cohere import CohereGenerator
        from cognity_ai.config.providers import CohereConfig
        return CohereGenerator(CohereConfig())

    def test_generate_returns_string(self, gen):
        result = gen.generate("What is ML?", context="ML is machine learning.")
        assert isinstance(result, str)


# ── AWS Bedrock Generator ─────────────────────────────────────────────────────

@requires_aws
class TestBedrockGenerator:
    @pytest.fixture
    def gen(self):
        pytest.importorskip("boto3")
        from cognity_ai.generators.bedrock import BedrockGenerator
        from cognity_ai.config.providers import BedrockConfig
        return BedrockGenerator(BedrockConfig())

    def test_generate_returns_string(self, gen):
        result = gen.generate("Hello", context="Say hi back.")
        assert isinstance(result, str)


# ── Azure OpenAI Generator ────────────────────────────────────────────────────

@requires_azure_openai
class TestAzureOpenAIGenerator:
    @pytest.fixture
    def gen(self):
        pytest.importorskip("openai")
        from cognity_ai.generators.azure_openai import AzureOpenAIGenerator
        from cognity_ai.config.providers import AzureOpenAIConfig
        return AzureOpenAIGenerator(AzureOpenAIConfig())

    def test_generate_returns_string(self, gen):
        result = gen.generate("Test", context="Test context.")
        assert isinstance(result, str)
