"""
raglib.multimodal.retrievers.cross_modal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-modal retriever using a unified embedding space (e.g., ImageBind).

Enables any-to-any retrieval:
  - Text query  → find relevant images, videos, AND audio segments
  - Image query → find related text, audio
  - Audio query → find related images, video segments

Requires an embedder that supports multiple modalities in a SHARED space
(ImageBind is the primary option; CLIP works for text↔image only).

Best used with ImageBindEmbedder.

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from raglib.multimodal.models.media import MultimodalRetrievalResult
from raglib.multimodal.retrievers.base import BaseMultimodalRetriever
from raglib.utils.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from raglib.generators.base import BaseGenerator
    from raglib.multimodal.embedders.base import BaseMultimodalEmbedder
    from raglib.multimodal.stores.base import BaseMultimodalStore

# Default set of modalities searched when the user does not restrict them.
_ALL_MODALITIES: list[str] = ["image", "video", "audio"]

# Mapping from modality name to the store method used to query it.
_STORE_METHOD: dict[str, str] = {
    "image": "query_images",
    "video": "query_videos",
    "audio": "query_audio",
}


class CrossModalRetriever(BaseMultimodalRetriever):
    """Cross-modal retriever that searches multiple modalities simultaneously.

    Uses a shared embedding space (e.g. ImageBind) to project any query
    modality into a common representation that can be compared against image,
    video, and audio indexes in a single retrieval pass.

    Results from the individual per-modality searches are merged with
    Reciprocal Rank Fusion (RRF) to produce a single ranked list that
    balances relevance signals across modalities.

    Parameters
    ----------
    embedder:
        A multimodal embedder whose embedding space is shared across all
        requested modalities (e.g.
        :class:`~raglib.multimodal.embedders.imagebind.ImageBindEmbedder`).
        CLIP-based embedders are supported for text↔image retrieval only.
    store:
        A multimodal vector store that implements ``query_images``,
        ``query_videos``, and/or ``query_audio`` depending on the configured
        *search_modalities*.
    generator:
        Optional text generator used to synthesise a fluent answer from the
        merged retrieval context.  When ``None``, :meth:`query` returns a
        structured plain-text summary instead.
    search_modalities:
        List of modality names to search.  Defaults to
        ``["image", "video", "audio"]``.  Pass a subset to restrict retrieval
        (e.g. ``["image", "video"]`` if no audio index is available).
    """

    def __init__(
        self,
        embedder: "BaseMultimodalEmbedder",
        store: "BaseMultimodalStore",
        generator: Optional["BaseGenerator"] = None,
        search_modalities: Optional[list[str]] = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._search_modalities: list[str] = (
            search_modalities if search_modalities is not None else list(_ALL_MODALITIES)
        )

        # Warn if the embedder's supported_modalities does not cover all
        # requested search modalities.
        supported: list[str] = getattr(embedder, "supported_modalities", []) or []
        missing = [m for m in self._search_modalities if m not in supported]
        if missing:
            warnings.warn(
                f"{type(embedder).__name__} does not declare support for "
                f"modalities {missing!r}. Cross-modal retrieval for those "
                "modalities may produce suboptimal results.  Consider using "
                "ImageBindEmbedder for full cross-modal support.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    @property
    def modality(self) -> str:
        """Returns ``"cross_modal"``."""
        return "cross_modal"

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve relevant content across all configured modalities.

        Embeds *query* as text, then fans out to each configured modality's
        store query method.  Per-modality ranked lists are merged using
        Reciprocal Rank Fusion and the fused top-*top_k* results are returned.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results to return after fusion.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Fused and re-ranked results across all modalities, tagged with
            their individual ``modality`` values (``"image"``, ``"video"``,
            ``"audio"``), sorted by RRF score descending.
        """
        embedding: list[float] = self._embedder.embed_text(query)
        return self._search_all_modalities(embedding, top_k)

    def retrieve_by_image(
        self,
        image: Union[str, bytes, Path],
        top_k: int = 5,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve cross-modal content similar to a query image.

        Embeds *image* using the image encoder and searches all configured
        modalities.  Works best with ImageBind, which maps image embeddings
        into the same space as text, audio, and video embeddings.

        Parameters
        ----------
        image:
            Query image as a filesystem path (``str`` or
            :class:`pathlib.Path`) or raw image bytes.
        top_k:
            Maximum number of results to return after fusion.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Fused and re-ranked results across all modalities.
        """
        embedding: list[float] = self._embedder.embed_image(image)
        return self._search_all_modalities(embedding, top_k)

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question by searching and synthesising across all modalities.

        Retrieves from all configured modalities and builds a rich context
        that labels each result by type and timestamp where applicable.
        If a generator is available, uses it to produce a fluent answer;
        otherwise returns the structured context directly.

        Parameters
        ----------
        question:
            Natural-language question.
        top_k:
            Maximum number of results to retrieve before building context.

        Returns
        -------
        str
            Generated or formatted answer, e.g.::

                [Image] A bar chart showing Q3 revenue figures.
                [Video @ 2:05 - 3:00] Discussion of budget projections...
                [Audio @ 0:45 - 1:10] CEO announces product launch date...
        """
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No relevant multimodal content found."

        context_parts: list[str] = []

        for result in results:
            mod = result.modality.upper()

            if result.modality == "image":
                chunk = result.image_chunk
                description = (
                    (chunk.caption if chunk else None)
                    or result.text
                    or "(no description)"
                )
                context_parts.append(f"[{mod}] {description}")

            elif result.modality == "video":
                chunk = result.video_chunk
                if chunk is not None:
                    start = _format_timestamp(int(chunk.start_ms))
                    end = _format_timestamp(int(chunk.end_ms))
                    text = chunk.transcript or chunk.scene_description or "(no description)"
                    context_parts.append(f"[{mod} @ {start} - {end}] {text}")
                else:
                    context_parts.append(f"[{mod}] {result.text or '(no description)'}")

            elif result.modality == "audio":
                chunk = result.audio_chunk
                if chunk is not None and chunk.segments:
                    first_seg = chunk.segments[0]
                    last_seg = chunk.segments[-1]
                    start = _format_timestamp(int(first_seg.start_ms))
                    end = _format_timestamp(int(last_seg.end_ms))
                    text = chunk.full_transcript or result.text or "(no transcript)"
                    context_parts.append(f"[{mod} @ {start} - {end}] {text}")
                else:
                    transcript = (
                        (chunk.full_transcript if chunk else None)
                        or result.text
                        or "(no transcript)"
                    )
                    context_parts.append(f"[{mod}] {transcript}")

            else:
                # Generic fallback for any additional future modalities.
                context_parts.append(f"[{mod}] {result.text or '(no description)'}")

        context = "\n".join(context_parts)

        if self._generator is not None:
            return self._generator.generate(question=question, context=context)

        return context

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_all_modalities(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[MultimodalRetrievalResult]:
        """Query each configured modality and fuse the results with RRF.

        Attempts to query each modality sequentially.  If a store method is
        missing or raises, that modality is silently skipped with a warning so
        that the remaining modalities can still return results.

        Parameters
        ----------
        embedding:
            Pre-computed query embedding in the shared embedding space.
        top_k:
            Number of results to request per modality before fusion.

        Returns
        -------
        list[MultimodalRetrievalResult]
            RRF-fused results, truncated to *top_k*.
        """
        per_modality_lists: list[list[MultimodalRetrievalResult]] = []

        for modality in self._search_modalities:
            method_name = _STORE_METHOD.get(modality)
            if method_name is None:
                warnings.warn(
                    f"Unknown modality {modality!r} — skipping.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            store_method = getattr(self._store, method_name, None)
            if store_method is None:
                warnings.warn(
                    f"Store {type(self._store).__name__!r} does not implement "
                    f"{method_name!r} — skipping modality {modality!r}.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            try:
                results: list[MultimodalRetrievalResult] = store_method(
                    embedding, top_k
                )
                # Tag each result with its modality before fusion.
                tagged = [
                    r.model_copy(update={"modality": modality}) for r in results
                ]
                per_modality_lists.append(tagged)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Error querying modality {modality!r}: {exc} — skipping.",
                    UserWarning,
                    stacklevel=3,
                )

        if not per_modality_lists:
            return []

        if len(per_modality_lists) == 1:
            return per_modality_lists[0][:top_k]

        # RRF requires RetrievalResult objects; MultimodalRetrievalResult is a
        # Pydantic model with a compatible ``score`` field.  The utility
        # function is generic over any object with .score, .source, and
        # .content attributes.  We adapt by passing a thin wrapper.
        #
        # Rather than monkey-patching, we perform fusion manually here to
        # avoid coupling to the text-only RRF utility's field assumptions.
        fused = _multimodal_rrf(per_modality_lists, top_k=top_k)
        return fused[:top_k]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _multimodal_rrf(
    ranked_lists: list[list[MultimodalRetrievalResult]],
    k: int = 60,
    top_k: int = 10,
) -> list[MultimodalRetrievalResult]:
    """Reciprocal Rank Fusion over :class:`MultimodalRetrievalResult` lists.

    Score for each chunk: ``Σ 1 / (k + rank_i)`` across all lists in which
    it appears.  Chunks appearing in multiple modality results are boosted.

    Parameters
    ----------
    ranked_lists:
        Per-modality ranked result lists.
    k:
        RRF constant (default 60, standard value).
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[MultimodalRetrievalResult]
        Merged and re-ranked list, sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, MultimodalRetrievalResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            key = result.chunk_id
            rrf_score = 1.0 / (k + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = result

    # Apply fused scores.
    fused = [
        result_map[key].model_copy(update={"score": scores[key]})
        for key in result_map
    ]
    return sorted(fused, key=lambda r: r.score, reverse=True)[:top_k]


def _format_timestamp(ms: int) -> str:
    """Convert a millisecond offset to a human-readable timestamp string.

    Parameters
    ----------
    ms:
        Time position in milliseconds (non-negative integer).

    Returns
    -------
    str
        Formatted timestamp, e.g. ``"2:05"`` or ``"1:02:30"``.
    """
    total_seconds = max(0, ms) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
