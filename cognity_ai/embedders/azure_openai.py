"""AzureOpenAIEmbedder — embedder using Azure OpenAI Service."""
from cognity_ai.embedders.base import BaseEmbedder


class AzureOpenAIEmbedder(BaseEmbedder):
    """Embed texts using an Azure OpenAI deployment.

    The AzureOpenAI client is created lazily; a fresh client is obtained per
    embed_batch call because the deployment name acts as the model identifier.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-02-01",
        dimensions: int = 1536,
        batch_size: int = 100,
    ):
        self._endpoint = endpoint
        self._api_key = api_key
        self._deployment = deployment
        self._api_version = api_version
        self._dims = dimensions
        self._batch_size = batch_size

    def _get_client(self):
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        client = self._get_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = client.embeddings.create(model=self._deployment, input=batch)
            results.extend([item.embedding for item in response.data])
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return self._dims
