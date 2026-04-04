"""Base observer ABC for the observability system.

Third-party integrations (OpenTelemetry, Langfuse, Prometheus, etc.) should
subclass :class:`BaseObserver` and override whichever hooks they need.
"""
from __future__ import annotations

from abc import ABC

from cognity_ai.observability.models import GenerationEvent, RetrievalEvent, EmbedEvent


class BaseObserver(ABC):
    """Abstract base for observability observers.

    All methods have no-op defaults so subclasses only need to override the
    events they care about.  This ensures forward compatibility when new event
    types are added.
    """

    def on_generation(self, event: GenerationEvent) -> None:  # noqa: B027
        """Called after every LLM generation."""

    def on_retrieval(self, event: RetrievalEvent) -> None:  # noqa: B027
        """Called after every retrieval query."""

    def on_embed(self, event: EmbedEvent) -> None:  # noqa: B027
        """Called after every embedding batch."""
