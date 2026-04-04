"""Zero-overhead no-op observer — the default when observability is disabled."""
from __future__ import annotations

from cognity_ai.observability.base_observer import BaseObserver
from cognity_ai.observability.models import GenerationEvent, RetrievalEvent, EmbedEvent


class NoopObserver(BaseObserver):
    """Does nothing. Used as the default observer to avoid None checks."""

    def on_generation(self, event: GenerationEvent) -> None:
        pass

    def on_retrieval(self, event: RetrievalEvent) -> None:
        pass

    def on_embed(self, event: EmbedEvent) -> None:
        pass
