"""GeminiEmbedder — default embedder using Google Gemini text-embedding-004."""
import time

from cognity_ai.embedders.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """Embed texts using the Google Generative AI embedding API.

    Rate limiting is applied between API calls to stay within rpm_limit.
    Batches are automatically chunked to respect the batch_limit per request.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
        batch_limit: int = 100,
        rpm_limit: int = 15,
    ):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = model
        self._batch_limit = batch_limit
        self._rpm_limit = rpm_limit
        self._last_call = 0.0

    def _rate_limit(self):
        """Ensure minimum gap between API calls to honour rpm_limit."""
        gap = 60.0 / self._rpm_limit
        elapsed = time.time() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.time()

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        """Batch embed with automatic chunking to respect API limits."""
        import google.generativeai as genai

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_limit):
            batch = texts[i : i + self._batch_limit]
            self._rate_limit()
            result = genai.embed_content(
                model=self._model,
                content=batch,
                task_type=task_type,
            )
            all_embeddings.extend(result["embedding"])
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with retrieval_query task type."""
        import google.generativeai as genai

        result = genai.embed_content(
            model=self._model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]

    @property
    def dimensions(self) -> int:
        return 768
