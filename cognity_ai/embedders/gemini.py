"""GeminiEmbedder — default embedder using Google Gemini text-embedding-004."""
import os
import time

from cognity_ai.embedders.base import BaseEmbedder


def _build_client(config_or_key=None, *, api_key=None, project_id=None,
                  location="us-central1", use_vertexai=False, timeout=120):
    """Build a google.genai Client from a GeminiConfig, an api_key string, or env vars."""
    from google import genai
    from google.genai import types as gentypes

    http_opts = gentypes.HttpOptions(timeout=timeout)

    # Config object passed
    if config_or_key is not None and not isinstance(config_or_key, str):
        cfg = config_or_key
        if getattr(cfg, "use_vertexai", False) and getattr(cfg, "project_id", ""):
            return genai.Client(
                vertexai=True,
                project=cfg.project_id,
                location=getattr(cfg, "location", "us-central1"),
                http_options=http_opts,
            )
        key = getattr(cfg, "api_key", "") or None
        if key:
            return genai.Client(api_key=key, http_options=http_opts)
        return genai.Client(http_options=http_opts)

    # Bare string or explicit api_key kwarg
    if isinstance(config_or_key, str):
        api_key = api_key or config_or_key

    if use_vertexai and project_id:
        return genai.Client(vertexai=True, project=project_id,
                            location=location, http_options=http_opts)
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_opts)
    # Auto-load GOOGLE_API_KEY from environment
    return genai.Client(http_options=http_opts)


class GeminiEmbedder(BaseEmbedder):
    """Embed texts using the Google Gemini embedding API (google-genai SDK).

    Accepts a ``GeminiConfig`` object (as used by the factory), an explicit
    ``api_key`` string, or no arguments at all — in which case the SDK
    automatically reads ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) from the
    environment.

    Examples::

        # Factory / config-object usage:
        embedder = GeminiEmbedder(cfg.gemini)

        # Explicit key:
        embedder = GeminiEmbedder(api_key="AIza...")

        # Env-var auto-load:
        embedder = GeminiEmbedder()
    """

    def __init__(
        self,
        config=None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        batch_limit: int | None = None,
        rpm_limit: int | None = None,
    ):
        # Resolve settings: config object takes precedence, kwargs override.
        cfg_model = "models/text-embedding-004"
        cfg_batch = 100
        cfg_rpm = 15
        cfg_timeout = 120

        if config is not None and not isinstance(config, str):
            cfg_model = getattr(config, "embedding_model", cfg_model)
            cfg_batch = getattr(config, "batch_embed_limit", cfg_batch)
            cfg_rpm = getattr(config, "rpm_limit", cfg_rpm)
            cfg_timeout = getattr(config, "timeout", cfg_timeout)
        elif isinstance(config, str):
            api_key = api_key or config
            config = None

        self._model = model or cfg_model
        self._batch_limit = batch_limit if batch_limit is not None else cfg_batch
        self._rpm_limit = rpm_limit if rpm_limit is not None else cfg_rpm
        self._last_call = 0.0
        self._client = _build_client(config, api_key=api_key, timeout=cfg_timeout)

    def _rate_limit(self):
        gap = 60.0 / self._rpm_limit
        elapsed = time.time() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.time()

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        """Batch embed with automatic chunking to respect API limits."""
        from google.genai import types as gentypes

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_limit):
            batch = texts[i : i + self._batch_limit]
            self._rate_limit()
            result = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=gentypes.EmbedContentConfig(task_type=task_type),
            )
            all_embeddings.extend(e.values for e in result.embeddings)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with retrieval_query task type."""
        from google.genai import types as gentypes

        self._rate_limit()
        result = self._client.models.embed_content(
            model=self._model,
            contents=query,
            config=gentypes.EmbedContentConfig(task_type="retrieval_query"),
        )
        return result.embeddings[0].values

    @property
    def dimensions(self) -> int:
        return 768
