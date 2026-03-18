"""CohereEmbedder — embedder using the Cohere embeddings API."""
from raglib.embedders.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """Embed texts using Cohere's embed endpoint.

    The task_type is mapped to Cohere's input_type convention:
      "retrieval_query"    → "search_query"
      anything else        → "search_document"
    """

    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        self._api_key = api_key
        self._model = model

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        import cohere

        client = cohere.Client(self._api_key)
        input_type = (
            "search_query" if task_type == "retrieval_query" else "search_document"
        )
        response = client.embed(
            texts=texts, model=self._model, input_type=input_type
        )
        return response.embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query], task_type="retrieval_query")[0]

    @property
    def dimensions(self) -> int:
        return 1024
