"""Token tracking with a priority chain: native → tiktoken → estimate.

Priority:
1. NativeTokenCounter  — extracted directly from API response objects
2. TiktokenCounter     — uses tiktoken; only when installed and model is known
3. EstimateCounter     — word-count heuristic (always available)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cognity_ai.observability.models import TokenUsage


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseTokenCounter(ABC):
    """ABC for pluggable token counters."""

    @abstractmethod
    def count(self, text: str, model: str = "") -> int:
        """Count tokens in *text* for *model*. Returns -1 when unavailable."""

    @abstractmethod
    def available(self) -> bool:
        """Return True when this counter can be used."""


# ── Concrete implementations ──────────────────────────────────────────────────

class NativeTokenCounter(BaseTokenCounter):
    """Extracts token counts from a raw API response object.

    This counter is always "available" in the sense that it will try to extract;
    if the response has no usage info, it returns -1 and the chain falls through.
    """

    # Maps provider name → extraction callable
    _EXTRACTORS: dict[str, Any] = {}

    def available(self) -> bool:
        return True

    def count(self, text: str, model: str = "") -> int:
        # Not applicable for plain text — use extract_from_response instead
        return -1

    def extract_from_response(self, response: Any, provider: str) -> TokenUsage:
        """Extract TokenUsage from a raw provider response object."""
        provider = provider.lower()
        try:
            if provider == "gemini":
                return self._from_gemini(response)
            if provider == "openai":
                return self._from_openai(response)
            if provider == "anthropic":
                return self._from_anthropic(response)
            if provider == "cohere":
                return self._from_cohere(response)
            if provider == "ollama":
                return self._from_ollama(response)
            if provider in ("bedrock", "aws"):
                return self._from_bedrock(response)
            if provider == "azure_openai":
                return self._from_openai(response)  # same structure
        except Exception:
            pass
        return TokenUsage(source="estimate")

    # ── Per-provider extractors ───────────────────────────────────────────────

    @staticmethod
    def _from_gemini(resp: Any) -> TokenUsage:
        usage = resp.usage_metadata
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
            source="native",
        )

    @staticmethod
    def _from_openai(resp: Any) -> TokenUsage:
        usage = resp.usage
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            source="native",
        )

    @staticmethod
    def _from_anthropic(message: Any) -> TokenUsage:
        usage = message.usage
        prompt = getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "output_tokens", 0) or 0
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            source="native",
        )

    @staticmethod
    def _from_cohere(resp: Any) -> TokenUsage:
        try:
            tokens = resp.meta.tokens
            prompt = getattr(tokens, "input_tokens", 0) or 0
            completion = getattr(tokens, "output_tokens", 0) or 0
            return TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                source="native",
            )
        except AttributeError:
            return TokenUsage(source="estimate")

    @staticmethod
    def _from_ollama(resp: Any) -> TokenUsage:
        # resp may be a requests.Response or a dict
        if hasattr(resp, "json"):
            data = resp.json()
        elif isinstance(resp, dict):
            data = resp
        else:
            return TokenUsage(source="estimate")
        prompt = data.get("prompt_eval_count", 0) or 0
        completion = data.get("eval_count", 0) or 0
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            source="native",
        )

    @staticmethod
    def _from_bedrock(resp: Any) -> TokenUsage:
        # Bedrock returns a StreamingBody; callers pass the decoded body dict
        if isinstance(resp, dict):
            prompt = resp.get("inputTokenCount", 0) or 0
            completion = resp.get("outputTokenCount", 0) or 0
            return TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                source="native",
            )
        return TokenUsage(source="estimate")


class TiktokenCounter(BaseTokenCounter):
    """Uses tiktoken when installed and the model is known to it."""

    def __init__(self) -> None:
        self._tiktoken: Any = None
        try:
            import tiktoken  # noqa: F401
            self._tiktoken = tiktoken
        except ImportError:
            pass

    def available(self) -> bool:
        return self._tiktoken is not None

    def count(self, text: str, model: str = "") -> int:
        if not self.available():
            return -1
        try:
            enc = self._tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not known to tiktoken — fall through
            return -1
        return len(enc.encode(text))


class EstimateCounter(BaseTokenCounter):
    """Simple word-count heuristic; always available."""

    def available(self) -> bool:
        return True

    def count(self, text: str, model: str = "") -> int:
        return max(1, len(text.split()))


# ── TokenTracker ──────────────────────────────────────────────────────────────

class TokenTracker:
    """Resolves the best available counter and extracts usage from responses.

    The priority chain is:
    1. :class:`NativeTokenCounter` (extracted from response object)
    2. :class:`TiktokenCounter` (if tiktoken installed and model known)
    3. :class:`EstimateCounter` (word-count fallback, always succeeds)

    Additional counters can be injected via *extra_counters*.
    """

    def __init__(self, extra_counters: list[BaseTokenCounter] | None = None) -> None:
        self._native = NativeTokenCounter()
        self._chain: list[BaseTokenCounter] = [
            TiktokenCounter(),
            EstimateCounter(),
        ]
        if extra_counters:
            # Insert before EstimateCounter so they get priority
            self._chain = extra_counters + self._chain

    def extract_from_response(self, response: Any, provider: str) -> TokenUsage:
        """Extract TokenUsage from a raw response object (uses native extractor)."""
        usage = self._native.extract_from_response(response, provider)
        if usage.source == "native":
            return usage
        # Native extraction failed — fall back to estimate on the response text
        return TokenUsage(source="estimate")

    def count_text(self, text: str, model: str = "") -> int:
        """Count tokens in *text* using the best available counter."""
        for counter in self._chain:
            if counter.available():
                result = counter.count(text, model)
                if result >= 0:
                    return result
        return max(1, len(text.split()))

    def build_usage_from_texts(
        self,
        prompt: str,
        completion: str,
        model: str = "",
    ) -> TokenUsage:
        """Build a :class:`TokenUsage` by counting prompt and completion texts."""
        source = "estimate"
        for counter in self._chain:
            if counter.available():
                p = counter.count(prompt, model)
                c = counter.count(completion, model)
                if p >= 0 and c >= 0:
                    if isinstance(counter, TiktokenCounter):
                        source = "tiktoken"
                    return TokenUsage(
                        prompt_tokens=p,
                        completion_tokens=c,
                        total_tokens=p + c,
                        source=source,
                    )
        p = max(1, len(prompt.split()))
        c = max(1, len(completion.split()))
        return TokenUsage(
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
            source="estimate",
        )
