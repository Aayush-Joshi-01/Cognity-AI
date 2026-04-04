"""AnthropicGenerator — generator using the Anthropic Messages API."""
from __future__ import annotations

import time

from cognity_ai.generators.base import BaseGenerator
from cognity_ai.observability.token_tracker import NativeTokenCounter

_NATIVE = NativeTokenCounter()


class AnthropicGenerator(BaseGenerator):
    """Generate answers using Anthropic Claude models.

    The anthropic client is created lazily to avoid importing the package
    at module load time.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, question: str, context: str) -> str:
        import anthropic
        from cognity_ai.observability.models import GenerationEvent

        client = anthropic.Anthropic(api_key=self._api_key)

        if question:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            # Pre-built prompt passed via generate_with_structured_context
            user_content = context

        t0 = time.time()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=(
                "You are a helpful assistant. "
                "Use the provided context to answer questions accurately."
            ),
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = (time.time() - t0) * 1000
        answer = message.content[0].text
        self._emit_generation(GenerationEvent(
            provider="anthropic",
            model=self._model,
            question=question,
            answer_length=len(answer),
            token_usage=_NATIVE.extract_from_response(message, "anthropic"),
            latency_ms=latency_ms,
        ))
        return answer
