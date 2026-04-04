"""Tests for the observability package.

No API keys required — all tests use mocks, NoopObserver, and LoggingObserver.
"""
from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from cognity_ai.observability.models import (
    TokenUsage,
    GenerationEvent,
    RetrievalEvent,
    EmbedEvent,
)
from cognity_ai.observability.base_observer import BaseObserver
from cognity_ai.observability.noop_observer import NoopObserver
from cognity_ai.observability.logging_observer import LoggingObserver
from cognity_ai.observability.collector import ObservabilityCollector
from cognity_ai.observability.token_tracker import (
    TokenTracker,
    EstimateCounter,
    TiktokenCounter,
    NativeTokenCounter,
)


# ── TokenUsage model ──────────────────────────────────────────────────────────

class TestTokenUsage:
    def test_defaults_to_zero(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.source == "native"

    def test_add(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, source="native")
        b = TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30, source="native")
        c = a + b
        assert c.prompt_tokens == 30
        assert c.completion_tokens == 15
        assert c.total_tokens == 45
        assert c.source == "native"

    def test_add_mixed_source(self):
        a = TokenUsage(source="native")
        b = TokenUsage(source="tiktoken")
        c = a + b
        assert c.source == "mixed"


# ── Event models ──────────────────────────────────────────────────────────────

class TestEventModels:
    def test_generation_event_defaults(self):
        ev = GenerationEvent(
            provider="gemini",
            model="gemini-2.0-flash",
            question="Q?",
            answer_length=42,
            latency_ms=120.5,
        )
        assert ev.event_type == "generation"
        assert isinstance(ev.token_usage, TokenUsage)
        assert isinstance(ev.timestamp, datetime)

    def test_retrieval_event(self):
        ev = RetrievalEvent(
            rag_method="vector_only",
            query="test query",
            top_k=5,
            results_count=3,
            latency_ms=50.0,
        )
        assert ev.event_type == "retrieval"
        assert ev.channels_used == []

    def test_embed_event(self):
        ev = EmbedEvent(
            provider="openai",
            model="text-embedding-3-small",
            batch_size=8,
            token_estimate=256,
            latency_ms=30.0,
        )
        assert ev.event_type == "embed"


# ── Token counters ────────────────────────────────────────────────────────────

class TestEstimateCounter:
    def test_always_available(self):
        assert EstimateCounter().available() is True

    def test_count_returns_word_count(self):
        ec = EstimateCounter()
        assert ec.count("hello world foo") == 3

    def test_count_empty_returns_one(self):
        assert EstimateCounter().count("") == 1


class TestTiktokenCounter:
    def test_available_only_when_installed(self):
        tc = TiktokenCounter()
        try:
            import tiktoken  # noqa: F401
            assert tc.available() is True
        except ImportError:
            assert tc.available() is False

    def test_count_unknown_model_returns_minus_one(self):
        tc = TiktokenCounter()
        if tc.available():
            assert tc.count("hello world", model="nonexistent-model-xyz") == -1

    def test_count_known_model(self):
        tc = TiktokenCounter()
        if tc.available():
            n = tc.count("hello world", model="gpt-4o")
            assert n > 0


class TestNativeTokenCounter:
    def test_always_available(self):
        assert NativeTokenCounter().available() is True

    def test_extract_gemini_response(self):
        mock_resp = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 5
        mock_resp.usage_metadata.total_token_count = 15
        counter = NativeTokenCounter()
        usage = counter.extract_from_response(mock_resp, "gemini")
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15
        assert usage.source == "native"

    def test_extract_openai_response(self):
        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = 20
        mock_resp.usage.completion_tokens = 10
        mock_resp.usage.total_tokens = 30
        counter = NativeTokenCounter()
        usage = counter.extract_from_response(mock_resp, "openai")
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 30

    def test_extract_anthropic_response(self):
        mock_msg = MagicMock()
        mock_msg.usage.input_tokens = 100
        mock_msg.usage.output_tokens = 50
        counter = NativeTokenCounter()
        usage = counter.extract_from_response(mock_msg, "anthropic")
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_extract_ollama_dict(self):
        data = {"prompt_eval_count": 8, "eval_count": 12}
        counter = NativeTokenCounter()
        usage = counter.extract_from_response(data, "ollama")
        assert usage.prompt_tokens == 8
        assert usage.completion_tokens == 12

    def test_extract_bedrock_dict(self):
        data = {"inputTokenCount": 50, "outputTokenCount": 25}
        counter = NativeTokenCounter()
        usage = counter.extract_from_response(data, "bedrock")
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 75

    def test_unknown_provider_falls_back(self):
        usage = NativeTokenCounter().extract_from_response(MagicMock(), "unknown_provider")
        assert usage.source == "estimate"


class TestTokenTracker:
    def test_count_text_returns_positive(self):
        tt = TokenTracker()
        assert tt.count_text("hello world") > 0

    def test_build_usage_from_texts(self):
        tt = TokenTracker()
        usage = tt.build_usage_from_texts("hello world", "foo bar baz")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_extract_from_gemini_response(self):
        mock_resp = MagicMock()
        mock_resp.usage_metadata.prompt_token_count = 7
        mock_resp.usage_metadata.candidates_token_count = 3
        mock_resp.usage_metadata.total_token_count = 10
        tt = TokenTracker()
        usage = tt.extract_from_response(mock_resp, "gemini")
        assert usage.source == "native"
        assert usage.total_tokens == 10


# ── Observers ─────────────────────────────────────────────────────────────────

class TestNoopObserver:
    def test_no_error_on_generation(self):
        ev = GenerationEvent(
            provider="x", model="m", question="q", answer_length=1, latency_ms=1.0
        )
        NoopObserver().on_generation(ev)  # must not raise

    def test_no_error_on_retrieval(self):
        ev = RetrievalEvent(rag_method="v", query="q", top_k=1, results_count=0, latency_ms=1.0)
        NoopObserver().on_retrieval(ev)

    def test_no_error_on_embed(self):
        ev = EmbedEvent(provider="x", model="m", batch_size=1, token_estimate=5, latency_ms=1.0)
        NoopObserver().on_embed(ev)


class TestLoggingObserver:
    def test_logs_generation_event(self, caplog):
        obs = LoggingObserver(level=logging.DEBUG)
        ev = GenerationEvent(
            provider="gemini", model="gemini-2.0-flash",
            question="Q", answer_length=10, latency_ms=50.0,
        )
        with caplog.at_level(logging.DEBUG, logger="cognity_ai.observability"):
            obs.on_generation(ev)
        assert len(caplog.records) >= 1
        assert "generation" in caplog.records[0].message

    def test_logs_retrieval_event(self, caplog):
        obs = LoggingObserver(level=logging.DEBUG)
        ev = RetrievalEvent(rag_method="v", query="q", top_k=5, results_count=3, latency_ms=10.0)
        with caplog.at_level(logging.DEBUG, logger="cognity_ai.observability"):
            obs.on_retrieval(ev)
        assert any("retrieval" in r.message for r in caplog.records)


# ── ObservabilityCollector ────────────────────────────────────────────────────

class TestObservabilityCollector:
    @pytest.fixture
    def collector(self):
        return ObservabilityCollector()

    def test_empty_summary(self, collector):
        s = collector.get_summary()
        assert s["total_generation_calls"] == 0
        assert s["total_tokens"] == 0

    def test_emit_generation_increments_count(self, collector):
        ev = GenerationEvent(
            provider="openai", model="gpt-4o", question="q",
            answer_length=5, latency_ms=100.0,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        collector.emit(ev)
        s = collector.get_summary()
        assert s["total_generation_calls"] == 1
        assert s["total_tokens"] == 15

    def test_emit_retrieval_increments_count(self, collector):
        ev = RetrievalEvent(rag_method="v", query="q", top_k=5, results_count=3, latency_ms=5.0)
        collector.emit(ev)
        assert collector.get_summary()["total_retrieval_calls"] == 1

    def test_emit_embed_increments_count(self, collector):
        ev = EmbedEvent(provider="x", model="m", batch_size=4, token_estimate=100, latency_ms=2.0)
        collector.emit(ev)
        assert collector.get_summary()["total_embed_calls"] == 1

    def test_recent_events(self, collector):
        for i in range(5):
            collector.emit(RetrievalEvent(
                rag_method="v", query=f"q{i}", top_k=1, results_count=1, latency_ms=1.0
            ))
        assert len(collector.recent_events(3)) == 3
        assert len(collector.recent_events(100)) == 5

    def test_reset_clears_stats(self, collector):
        ev = GenerationEvent(
            provider="x", model="m", question="q", answer_length=1, latency_ms=1.0,
            token_usage=TokenUsage(total_tokens=50),
        )
        collector.emit(ev)
        collector.reset()
        s = collector.get_summary()
        assert s["total_generation_calls"] == 0
        assert s["total_tokens"] == 0
        assert len(collector.recent_events()) == 0

    def test_observer_receives_event(self):
        received = []

        class RecordObserver(BaseObserver):
            def on_generation(self, event):
                received.append(event)

        c = ObservabilityCollector(observers=[RecordObserver()])
        ev = GenerationEvent(
            provider="x", model="m", question="q", answer_length=1, latency_ms=1.0
        )
        c.emit(ev)
        assert len(received) == 1
        assert received[0] is ev

    def test_disabled_collector_ignores_events(self):
        c = ObservabilityCollector(enabled=False)
        ev = GenerationEvent(
            provider="x", model="m", question="q", answer_length=1, latency_ms=1.0
        )
        c.emit(ev)
        assert c.get_summary()["total_generation_calls"] == 0

    def test_add_remove_observer(self):
        received = []

        class R(BaseObserver):
            def on_generation(self, event):
                received.append(event)

        obs = R()
        c = ObservabilityCollector()
        c.add_observer(obs)
        ev = GenerationEvent(provider="x", model="m", question="q", answer_length=1, latency_ms=1.0)
        c.emit(ev)
        assert len(received) == 1
        c.remove_observer(obs)
        c.emit(ev)
        assert len(received) == 1  # no new events after removal

    def test_ring_buffer_max_size(self):
        c = ObservabilityCollector(max_event_buffer=3)
        for i in range(5):
            c.emit(RetrievalEvent(
                rag_method="v", query=f"q{i}", top_k=1, results_count=1, latency_ms=1.0
            ))
        assert len(c.recent_events(100)) == 3

    def test_faulty_observer_does_not_crash_pipeline(self):
        class BrokenObserver(BaseObserver):
            def on_generation(self, event):
                raise RuntimeError("broken!")

        c = ObservabilityCollector(observers=[BrokenObserver()])
        ev = GenerationEvent(provider="x", model="m", question="q", answer_length=1, latency_ms=1.0)
        c.emit(ev)  # must not raise
        assert c.get_summary()["total_generation_calls"] == 1


# ── BaseGenerator integration ─────────────────────────────────────────────────

class TestGeneratorCollectorIntegration:
    def test_set_collector_and_emit(self):
        """Generator emits event to collector when set_collector is called."""
        pytest.importorskip("faiss")  # need faiss for MinimalLibraryConfig
        from cognity_ai.generators.base import BaseGenerator
        from cognity_ai.observability.models import GenerationEvent

        class StubGenerator(BaseGenerator):
            def generate(self, question: str, context: str) -> str:
                self._emit_generation(GenerationEvent(
                    provider="stub", model="stub-v1",
                    question=question, answer_length=4, latency_ms=1.0,
                ))
                return "test"

        gen = StubGenerator()
        collector = ObservabilityCollector()
        gen.set_collector(collector)
        gen.generate("Q?", "context")
        assert collector.get_summary()["total_generation_calls"] == 1
