"""Observer that logs events as structured JSON via Python's logging module."""
from __future__ import annotations

import json
import logging

from cognity_ai.observability.base_observer import BaseObserver
from cognity_ai.observability.models import GenerationEvent, RetrievalEvent, EmbedEvent

_LOGGER_NAME = "cognity_ai.observability"


class LoggingObserver(BaseObserver):
    """Emits each event as a JSON-serialised log record.

    Args:
        level: Python logging level (default: ``logging.INFO``).
        logger_name: Name of the logger to use.
    """

    def __init__(
        self,
        level: int = logging.INFO,
        logger_name: str = _LOGGER_NAME,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def on_generation(self, event: GenerationEvent) -> None:
        self._emit(event.model_dump())

    def on_retrieval(self, event: RetrievalEvent) -> None:
        self._emit(event.model_dump())

    def on_embed(self, event: EmbedEvent) -> None:
        self._emit(event.model_dump())

    def _emit(self, data: dict) -> None:
        try:
            self._logger.log(self._level, json.dumps(data, default=str))
        except Exception:
            pass
