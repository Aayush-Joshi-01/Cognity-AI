"""OllamaGenerator — generator using a locally running Ollama server."""
from __future__ import annotations

import time

from cognity_ai.generators.base import BaseGenerator
from cognity_ai.observability.token_tracker import NativeTokenCounter

_NATIVE = NativeTokenCounter()


class OllamaGenerator(BaseGenerator):
    """Generate answers via the Ollama REST API (/api/chat).

    Requires a running Ollama instance (default: http://localhost:11434).
    Streaming is disabled so the full response is returned as a single string.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature

    def generate(self, question: str, context: str) -> str:
        import requests

        if question:
            system_content = f"Context: {context}"
            user_content = question
        else:
            # Pre-built prompt passed via generate_with_structured_context
            system_content = ""
            user_content = context

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        from cognity_ai.observability.models import GenerationEvent
        t0 = time.time()
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self._temperature},
            },
        )
        resp.raise_for_status()
        latency_ms = (time.time() - t0) * 1000
        data = resp.json()
        answer = data["message"]["content"]
        self._emit_generation(GenerationEvent(
            provider="ollama",
            model=self._model,
            question=question,
            answer_length=len(answer),
            token_usage=_NATIVE.extract_from_response(data, "ollama"),
            latency_ms=latency_ms,
        ))
        return answer
