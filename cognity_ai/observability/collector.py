"""ObservabilityCollector — fan-out hub with an in-memory event ring buffer."""
from __future__ import annotations

from collections import deque
from typing import Union

from cognity_ai.observability.base_observer import BaseObserver
from cognity_ai.observability.noop_observer import NoopObserver
from cognity_ai.observability.models import (
    GenerationEvent,
    RetrievalEvent,
    EmbedEvent,
    TokenUsage,
)

_AnyEvent = Union[GenerationEvent, RetrievalEvent, EmbedEvent]


class ObservabilityCollector:
    """Receives events and fans them out to all registered observers.

    Also maintains:
    - Aggregate counters (total calls, total tokens, etc.)
    - A configurable-size ring buffer of recent events
    - A ``get_summary()`` method for quick cost/usage reporting

    Args:
        observers: Initial list of :class:`BaseObserver` instances.
        enabled: When False, all emits are no-ops.
        max_event_buffer: Maximum number of recent events to keep in memory.
    """

    def __init__(
        self,
        observers: list[BaseObserver] | None = None,
        enabled: bool = True,
        max_event_buffer: int = 1000,
    ) -> None:
        self._enabled = enabled
        self._observers: list[BaseObserver] = list(observers or [])
        self._buffer: deque[_AnyEvent] = deque(maxlen=max_event_buffer)

        # Aggregate stats
        self._generation_count = 0
        self._retrieval_count = 0
        self._embed_count = 0
        self._total_usage = TokenUsage()

    # ── Observer management ───────────────────────────────────────────────────

    def add_observer(self, observer: BaseObserver) -> None:
        """Register an additional observer."""
        self._observers.append(observer)

    def remove_observer(self, observer: BaseObserver) -> None:
        """Unregister an observer (no-op if not registered)."""
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    # ── Emit ──────────────────────────────────────────────────────────────────

    def emit(self, event: _AnyEvent) -> None:
        """Fan out *event* to all observers and update internal stats."""
        if not self._enabled:
            return
        self._buffer.append(event)
        self._update_stats(event)
        for obs in self._observers:
            try:
                if isinstance(event, GenerationEvent):
                    obs.on_generation(event)
                elif isinstance(event, RetrievalEvent):
                    obs.on_retrieval(event)
                elif isinstance(event, EmbedEvent):
                    obs.on_embed(event)
            except Exception:
                pass  # observers must never crash the pipeline

    # ── Query ─────────────────────────────────────────────────────────────────

    def recent_events(self, n: int = 100) -> list[_AnyEvent]:
        """Return the *n* most recent events (oldest first)."""
        events = list(self._buffer)
        return events[-n:] if n < len(events) else events

    def get_summary(self) -> dict:
        """Return aggregate usage statistics."""
        return {
            "enabled": self._enabled,
            "total_generation_calls": self._generation_count,
            "total_retrieval_calls": self._retrieval_count,
            "total_embed_calls": self._embed_count,
            "total_calls": self._generation_count + self._retrieval_count + self._embed_count,
            "total_prompt_tokens": self._total_usage.prompt_tokens,
            "total_completion_tokens": self._total_usage.completion_tokens,
            "total_tokens": self._total_usage.total_tokens,
            "token_count_source": self._total_usage.source,
            "buffered_events": len(self._buffer),
        }

    def reset(self) -> None:
        """Clear all counters and the event buffer."""
        self._generation_count = 0
        self._retrieval_count = 0
        self._embed_count = 0
        self._total_usage = TokenUsage()
        self._buffer.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_stats(self, event: _AnyEvent) -> None:
        if isinstance(event, GenerationEvent):
            self._generation_count += 1
            self._total_usage = self._total_usage + event.token_usage
        elif isinstance(event, RetrievalEvent):
            self._retrieval_count += 1
        elif isinstance(event, EmbedEvent):
            self._embed_count += 1
