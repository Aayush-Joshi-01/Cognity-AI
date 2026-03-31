"""SentenceTransformerEmbedder — fully local embedder using HuggingFace sentence-transformers."""
from cognity_ai.embedders.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embed texts locally using a sentence-transformers model.

    No API key or network access is required after the model is downloaded.
    The model is loaded lazily on first use.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _get_model(self):
        if not self._model:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()
