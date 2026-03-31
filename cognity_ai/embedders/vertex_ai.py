"""VertexAIEmbedder — embedder using Google Cloud Vertex AI text-embedding-005."""
from cognity_ai.embedders.base import BaseEmbedder


class VertexAIEmbedder(BaseEmbedder):
    """Embed texts using Vertex AI TextEmbeddingModel.

    Vertex AI allows up to 250 texts per get_embeddings() call; this class
    automatically chunks larger batches.
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "text-embedding-005",
    ):
        self._project = project
        self._location = location
        self._model_name = model
        self._model = None

    def _get_model(self):
        if not self._model:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(project=self._project, location=self._location)
            self._model = TextEmbeddingModel.from_pretrained(self._model_name)
        return self._model

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        """Embed texts in chunks of 250 (Vertex AI per-request limit)."""
        model = self._get_model()
        results: list[list[float]] = []
        for i in range(0, len(texts), 250):
            batch = texts[i : i + 250]
            embeddings = model.get_embeddings(batch)
            results.extend([e.values for e in embeddings])
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return 768
