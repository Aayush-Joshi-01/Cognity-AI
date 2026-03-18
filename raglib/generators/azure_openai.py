"""AzureOpenAIGenerator — generator using Azure OpenAI Service."""
from raglib.generators.base import BaseGenerator


class AzureOpenAIGenerator(BaseGenerator):
    """Generate answers using an Azure OpenAI deployment.

    The AzureOpenAI client is created lazily; a fresh client is obtained per
    generate() call (the client is lightweight and stateless).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-02-01",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self._endpoint = endpoint
        self._api_key = api_key
        self._deployment = deployment
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _get_client(self):
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )

    def generate(self, question: str, context: str) -> str:
        client = self._get_client()

        if question:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            # Pre-built prompt passed via generate_with_structured_context
            user_content = context

        resp = client.chat.completions.create(
            model=self._deployment,
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
