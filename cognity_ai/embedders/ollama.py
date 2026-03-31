"""OllamaEmbedder — embedder using a locally running Ollama server."""
from cognity_ai.embedders.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """Embed texts via the Ollama REST API (/api/embeddings).

    Requires a running Ollama instance (default: http://localhost:11434).
    Texts are embedded one at a time because the Ollama embeddings endpoint
    accepts a single prompt per request.

    The reported dimensions (768) are a sensible default for nomic-embed-text;
    the actual size depends on the chosen model.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        import requests

        results: list[list[float]] = []
        for text in texts:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return 768  # varies by model; nomic-embed-text default
