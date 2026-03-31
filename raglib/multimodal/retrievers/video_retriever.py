"""
raglib.multimodal.retrievers.video_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video retriever for multimodal RAG.

Finds relevant video segments (scenes) given a text query.
Returns segments with timestamps for navigation.

Example use case::

    retriever = VideoRetriever(embedder=embedder, store=store)
    results = retriever.retrieve("Find the part where they discuss the budget")
    # → Returns VideoChunk with start_ms=125000, end_ms=180000

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from raglib.multimodal.models.media import MultimodalRetrievalResult
from raglib.multimodal.retrievers.base import BaseMultimodalRetriever

if TYPE_CHECKING:
    from raglib.generators.base import BaseGenerator
    from raglib.multimodal.embedders.base import BaseMultimodalEmbedder
    from raglib.multimodal.stores.base import BaseMultimodalStore


class VideoRetriever(BaseMultimodalRetriever):
    """Retriever for video modality using frame/clip embeddings.

    Converts text queries into embedding vectors and searches an indexed
    collection of :class:`~raglib.multimodal.models.media.VideoChunk` objects,
    each representing a scene or temporal segment of a video.

    Results include timestamps (``start_ms`` / ``end_ms``) that can be used
    to seek directly to the relevant part of the video.

    Parameters
    ----------
    embedder:
        A multimodal embedder whose text encoder operates in the same
        embedding space as the indexed video chunks
        (e.g. CLIP, SigLIP, or ImageBind).
    store:
        A multimodal vector store that implements ``query_videos``.
    generator:
        Optional text generator used to synthesise a fluent answer from
        retrieved transcripts and scene descriptions.  When ``None``,
        :meth:`query` returns a formatted list of timestamped transcript
        excerpts instead.
    """

    def __init__(
        self,
        embedder: "BaseMultimodalEmbedder",
        store: "BaseMultimodalStore",
        generator: Optional["BaseGenerator"] = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    @property
    def modality(self) -> str:
        """Returns ``"video"``."""
        return "video"

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most relevant video segments for a text query.

        Embeds *query* using the shared CLIP/SigLIP/ImageBind text encoder
        and performs an approximate nearest-neighbour search over the video
        segment index.

        Parameters
        ----------
        query:
            Natural-language description of the desired video content
            (e.g. ``"budget discussion"``, ``"product demo"``,
            ``"interview with CEO"``).
        top_k:
            Maximum number of video segment results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of results with ``modality="video"``, most relevant
            first.  Each result's ``video_chunk`` field is populated with
            timestamp and transcript information.
        """
        embedding: list[float] = self._embedder.embed_text(query)
        raw_results: list[MultimodalRetrievalResult] = self._store.query_videos(
            embedding, top_k
        )
        return [
            result.model_copy(update={"modality": "video"})
            for result in raw_results
        ]

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question about video content using retrieved segments.

        Retrieves the most relevant video segments, then either generates a
        fluent answer (if a generator was provided) or returns a formatted
        list of timestamped transcript excerpts.

        Parameters
        ----------
        question:
            Natural-language question about video content.
        top_k:
            Maximum number of segments to retrieve.

        Returns
        -------
        str
            Generated or formatted answer, with timestamps in
            ``HH:MM:SS`` or ``MM:SS`` format for easy navigation.

        Example output (no generator)::

            At 0:05 - 0:12: Welcome to the Q3 earnings call.
            At 2:05 - 3:00: Our budget for next quarter is projected at...
            At 5:30 - 6:15: In summary, we expect strong growth in...
        """
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No relevant video segments found."

        context_parts: list[str] = []
        transcript_parts: list[str] = []  # kept separate for generator context

        for result in results:
            chunk = result.video_chunk
            if chunk is None:
                description = result.text or "(no description)"
                context_parts.append(f"Segment: {description}")
                transcript_parts.append(description)
                continue

            start_fmt = self.format_timestamp(int(chunk.start_ms))
            end_fmt = self.format_timestamp(int(chunk.end_ms))
            timestamp_label = f"At {start_fmt} - {end_fmt}"

            transcript = chunk.transcript or chunk.scene_description or "(no transcript)"
            context_parts.append(f"{timestamp_label}: {transcript}")
            transcript_parts.append(f"{timestamp_label}: {transcript}")

        context = "\n".join(context_parts)

        if self._generator is not None:
            return self._generator.generate(question=question, context=context)

        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_timestamp(ms: int) -> str:
        """Convert a millisecond offset to a human-readable timestamp string.

        Produces ``"H:MM:SS"`` for clips longer than one hour, or ``"M:SS"``
        otherwise (matching common video player conventions).

        Parameters
        ----------
        ms:
            Time position in milliseconds (non-negative integer).

        Returns
        -------
        str
            Formatted timestamp, e.g. ``"1:23:45"`` or ``"2:05"``.

        Examples
        --------
        >>> VideoRetriever.format_timestamp(0)
        '0:00'
        >>> VideoRetriever.format_timestamp(65000)
        '1:05'
        >>> VideoRetriever.format_timestamp(3725000)
        '1:02:05'
        """
        total_seconds = max(0, ms) // 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
