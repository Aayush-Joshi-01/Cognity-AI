"""
raglib.multimodal.retrievers.audio_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Audio retriever for multimodal RAG.

Finds relevant audio segments given a text query.
Uses transcript embeddings for semantic search.

Example use case::

    retriever = AudioRetriever(embedder=embedder, store=store)
    results = retriever.retrieve("Find where the speaker mentions product launch")
    # → Returns AudioSegment with start_ms, end_ms, transcript snippet

Retrieval strategy
------------------
1. If the configured *embedder* supports audio (e.g. ImageBind), the query
   text is embedded in the shared audio/text space.
2. Otherwise, if a *text_embedder* is provided, the query is embedded with it
   and matched against transcript-derived text embeddings stored in the index.
3. As a final fallback the primary *embedder*'s ``embed_text`` is used.

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

from raglib.multimodal.models.media import MultimodalRetrievalResult
from raglib.multimodal.retrievers.base import BaseMultimodalRetriever

if TYPE_CHECKING:
    from raglib.generators.base import BaseGenerator
    from raglib.multimodal.embedders.base import BaseMultimodalEmbedder
    from raglib.multimodal.stores.base import BaseMultimodalStore


class AudioRetriever(BaseMultimodalRetriever):
    """Retriever for audio modality using transcript or audio embeddings.

    Searches an indexed collection of
    :class:`~raglib.multimodal.models.media.AudioChunk` objects by embedding
    the text query and performing approximate nearest-neighbour search.

    Two embedding strategies are supported:

    1. **Audio-native** (ImageBind): the query text is projected into the
       shared audio/text embedding space, enabling direct semantic matching
       against audio embeddings.
    2. **Transcript-based** (all other embedders): the query is embedded with
       *text_embedder* (or *embedder* as a fallback) and matched against
       embeddings derived from the audio transcripts.

    Parameters
    ----------
    embedder:
        Multimodal embedder.  If it supports the ``"audio"`` modality
        (i.e. ``embed_audio`` does not raise), it is used for audio-native
        embedding.
    store:
        A multimodal vector store that implements ``query_audio``.
    generator:
        Optional text generator used to synthesise a fluent answer from
        retrieved transcripts.  When ``None``, :meth:`query` returns a
        formatted list of timestamped transcript excerpts.
    text_embedder:
        Optional dedicated text embedder (e.g. a sentence-transformer).
        When provided, this is preferred over *embedder* for transcript-based
        retrieval so that a more capable text model can be used alongside a
        multimodal audio model.
    """

    def __init__(
        self,
        embedder: "BaseMultimodalEmbedder",
        store: "BaseMultimodalStore",
        generator: Optional["BaseGenerator"] = None,
        text_embedder: Optional["BaseMultimodalEmbedder"] = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._text_embedder = text_embedder

        # Detect whether the primary embedder supports audio natively.
        self._audio_native: bool = "audio" in (
            getattr(embedder, "supported_modalities", []) or []
        )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    @property
    def modality(self) -> str:
        """Returns ``"audio"``."""
        return "audio"

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most relevant audio segments for a text query.

        Selects the embedding strategy automatically:

        * If the primary embedder supports audio natively (e.g. ImageBind),
          uses it to embed the text query in the shared audio/text space.
        * Otherwise falls back to the dedicated *text_embedder* if provided,
          or to the primary embedder's ``embed_text`` method.

        Parameters
        ----------
        query:
            Natural-language search query describing the desired audio content
            (e.g. ``"product launch announcement"``).
        top_k:
            Maximum number of audio segment results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of results with ``modality="audio"``, most relevant
            first.  Each result's ``audio_chunk`` field is populated with
            transcript and timestamp information.
        """
        embedding = self._embed_query(query)
        raw_results: list[MultimodalRetrievalResult] = self._store.query_audio(
            embedding, top_k
        )
        return [
            result.model_copy(update={"modality": "audio"})
            for result in raw_results
        ]

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question about audio content using retrieved segments.

        Retrieves the most relevant audio segments, then either generates a
        fluent answer (if a generator was provided) or returns a formatted
        list of timestamped transcript excerpts.

        Parameters
        ----------
        question:
            Natural-language question about audio content.
        top_k:
            Maximum number of segments to retrieve.

        Returns
        -------
        str
            Generated or formatted answer grounded in the retrieved audio
            transcripts, with timestamps for easy navigation.
        """
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No relevant audio segments found."

        context_parts: list[str] = []

        for result in results:
            chunk = result.audio_chunk
            if chunk is None:
                text = result.text or "(no transcript)"
                context_parts.append(f"Segment: {text}")
                continue

            # Prefer first populated segment for timestamp display.
            if chunk.segments:
                first_seg = chunk.segments[0]
                last_seg = chunk.segments[-1]
                start_fmt = _format_timestamp(int(first_seg.start_ms))
                end_fmt = _format_timestamp(int(last_seg.end_ms))
                timestamp_label = f"At {start_fmt} - {end_fmt}"
            else:
                timestamp_label = "Audio segment"

            transcript = chunk.full_transcript or result.text or "(no transcript)"
            context_parts.append(f"{timestamp_label}: {transcript}")

        context = "\n".join(context_parts)

        if self._generator is not None:
            return self._generator.generate(question=question, context=context)

        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """Return the most appropriate embedding for *query*.

        Selection order:

        1. Audio-native embedder (ImageBind) — projects text into the shared
           audio/text space for direct audio matching.
        2. Dedicated *text_embedder* — used when a higher-quality text model
           is configured alongside a multimodal embedder.
        3. Primary *embedder* ``embed_text`` — universal fallback.

        Parameters
        ----------
        query:
            The text query to embed.

        Returns
        -------
        list[float]
            Embedding vector for the query.
        """
        if self._audio_native:
            # embed_text on an audio-native model projects into shared space.
            return self._embedder.embed_text(query)

        if self._text_embedder is not None:
            return self._text_embedder.embed_text(query)

        # Fallback: primary embedder's text encoder.
        warnings.warn(
            f"{type(self._embedder).__name__} does not natively support audio "
            "embeddings.  Falling back to text-based transcript retrieval.  "
            "For better audio retrieval, use ImageBindEmbedder or supply a "
            "text_embedder.",
            UserWarning,
            stacklevel=3,
        )
        return self._embedder.embed_text(query)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _format_timestamp(ms: int) -> str:
    """Convert a millisecond offset to a human-readable timestamp string.

    Produces ``"H:MM:SS"`` for durations longer than one hour, or
    ``"M:SS"`` otherwise.

    Parameters
    ----------
    ms:
        Time position in milliseconds (non-negative integer).

    Returns
    -------
    str
        Formatted timestamp, e.g. ``"0:45"`` or ``"1:02:30"``.
    """
    total_seconds = max(0, ms) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
