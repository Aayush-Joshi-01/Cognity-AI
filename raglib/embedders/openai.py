"""OpenAIEmbedder — embedder using the OpenAI embeddings API."""
from raglib.embedders.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Embed texts using OpenAI's embeddings endpoint.

    The client is created lazily so that the openai package is only
    imported when actually used.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 100,
    ):
        self._api_key = api_key
        self._model = model
        self._dims = dimensions
        self._batch_size = batch_size
        self._client = None

    def _get_client(self):
        if not self._client:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        client = self._get_client()
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = client.embeddings.create(model=self._model, input=batch)
            results.extend([item.embedding for item in response.data])
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return self._dims
