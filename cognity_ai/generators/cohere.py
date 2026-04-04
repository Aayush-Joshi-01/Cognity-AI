"""CohereGenerator — generator using the Cohere Chat API."""
from __future__ import annotations

import time

from cognity_ai.generators.base import BaseGenerator
from cognity_ai.observability.token_tracker import NativeTokenCounter

_NATIVE = NativeTokenCounter()


class CohereGenerator(BaseGenerator):
    """Generate answers using Cohere's chat endpoint (command-r-plus or similar).

    The cohere client is created lazily to avoid importing the package at
    module load time.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-plus",
        temperature: float = 0.1,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def generate(self, question: str, context: str) -> str:
        import cohere

        client = cohere.Client(self._api_key)

        if question:
            message = question
            preamble = f"Use this context to answer: {context}"
        else:
            # Pre-built prompt passed via generate_with_structured_context;
            # treat it as the message with no separate preamble.
            message = context
            preamble = ""

        kwargs = {
            "model": self._model,
            "message": message,
            "temperature": self._temperature,
        }
        if preamble:
            kwargs["preamble"] = preamble

        from cognity_ai.observability.models import GenerationEvent
        t0 = time.time()
        resp = client.chat(**kwargs)
        latency_ms = (time.time() - t0) * 1000
        answer = resp.text
        self._emit_generation(GenerationEvent(
            provider="cohere",
            model=self._model,
            question=question,
            answer_length=len(answer),
            token_usage=_NATIVE.extract_from_response(resp, "cohere"),
            latency_ms=latency_ms,
        ))
        return answer
