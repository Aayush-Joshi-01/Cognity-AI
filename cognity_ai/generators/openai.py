"""OpenAIGenerator — generator using the OpenAI Chat Completions API."""
from cognity_ai.generators.base import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    """Generate answers using OpenAI's chat completions endpoint.

    The OpenAI client is created lazily to avoid importing the package at
    module load time.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, question: str, context: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)

        if question:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            # Pre-built prompt passed via generate_with_structured_context
            user_content = context

        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Use the provided context to answer questions accurately."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content
