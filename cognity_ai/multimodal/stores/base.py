"""
raglib.multimodal.stores.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for multimodal vector store backends.

All concrete store implementations (ChromaDB, Qdrant, …) must subclass
:class:`BaseMultimodalStore` and implement every abstract method.

.. warning::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from cognity_ai.multimodal.models.media import (
    AudioChunk,
    ImageChunk,
    MultimodalRetrievalResult,
    VideoChunk,
)


class BaseMultimodalStore(ABC):
    """Abstract base class for multimodal vector store backends.

    A multimodal store manages three distinct modality collections — images,
    videos, and audio — each with their own embedding space and metadata
    schema.  Implementations are free to use a single physical database with
    per-modality namespacing (e.g. separate Chroma collections or Qdrant
    collection names) or a unified collection with a ``modality`` payload
    field.

    All query methods accept an optional ``filters`` dict whose keys are
    backend-specific (e.g. ``{"doc_id": "abc123"}``); implementations should
    document which filter keys they honour.
    """

    # ── Write operations ─────────────────────────────────────────────────

    @abstractmethod
    def upsert_image(self, chunk: ImageChunk) -> None:
        """Store or update an :class:`~raglib.multimodal.models.media.ImageChunk`.

        The chunk's ``embedding`` must be populated before calling this method.
        Image bytes are intentionally *not* stored inside the vector backend;
        implementations should persist ``chunk.image_path`` (or equivalent)
        as a metadata field for later retrieval.

        Args:
            chunk: The image chunk to upsert, including its dense embedding.
        """
        ...

    @abstractmethod
    def upsert_video(self, chunk: VideoChunk) -> None:
        """Store or update a :class:`~raglib.multimodal.models.media.VideoChunk`.

        The chunk's ``embedding`` (typically a mean-pool of frame embeddings)
        must be populated before calling this method.

        Args:
            chunk: The video chunk to upsert, including its dense embedding.
        """
        ...

    @abstractmethod
    def upsert_audio(self, chunk: AudioChunk) -> None:
        """Store or update an :class:`~raglib.multimodal.models.media.AudioChunk`.

        The chunk's ``embedding`` must be populated before calling this method.

        Args:
            chunk: The audio chunk to upsert, including its dense embedding.
        """
        ...

    # ── Query operations ─────────────────────────────────────────────────

    @abstractmethod
    def query_images(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most similar image chunks for the given query embedding.

        Args:
            embedding: Dense query vector in the same space as the stored
                image embeddings.
            top_k: Maximum number of results to return.
            filters: Optional backend-specific filter dict (e.g.
                ``{"doc_id": "abc123"}``).

        Returns:
            A list of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            objects ordered by descending relevance score, with
            ``modality="image"`` and the ``image_chunk`` field populated.
        """
        ...

    @abstractmethod
    def query_videos(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most similar video chunks for the given query embedding.

        Args:
            embedding: Dense query vector in the same space as the stored
                video embeddings.
            top_k: Maximum number of results to return.
            filters: Optional backend-specific filter dict.

        Returns:
            A list of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            objects ordered by descending relevance score, with
            ``modality="video"`` and the ``video_chunk`` field populated.
        """
        ...

    @abstractmethod
    def query_audio(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most similar audio chunks for the given query embedding.

        Args:
            embedding: Dense query vector in the same space as the stored
                audio embeddings.
            top_k: Maximum number of results to return.
            filters: Optional backend-specific filter dict.

        Returns:
            A list of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            objects ordered by descending relevance score, with
            ``modality="audio"`` and the ``audio_chunk`` field populated.
        """
        ...

    # ── Point-retrieval by ID ────────────────────────────────────────────

    @abstractmethod
    def get_image_by_id(self, chunk_id: str) -> Optional[ImageChunk]:
        """Fetch a single :class:`~raglib.multimodal.models.media.ImageChunk` by its ID.

        Args:
            chunk_id: The unique identifier of the image chunk.

        Returns:
            The matching :class:`~raglib.multimodal.models.media.ImageChunk`,
            or ``None`` if not found.
        """
        ...

    @abstractmethod
    def get_video_by_id(self, chunk_id: str) -> Optional[VideoChunk]:
        """Fetch a single :class:`~raglib.multimodal.models.media.VideoChunk` by its ID.

        Args:
            chunk_id: The unique identifier of the video chunk.

        Returns:
            The matching :class:`~raglib.multimodal.models.media.VideoChunk`,
            or ``None`` if not found.
        """
        ...

    @abstractmethod
    def get_audio_by_id(self, chunk_id: str) -> Optional[AudioChunk]:
        """Fetch a single :class:`~raglib.multimodal.models.media.AudioChunk` by its ID.

        Args:
            chunk_id: The unique identifier of the audio chunk.

        Returns:
            The matching :class:`~raglib.multimodal.models.media.AudioChunk`,
            or ``None`` if not found.
        """
        ...

    # ── Deletion ─────────────────────────────────────────────────────────

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all stored chunks (across all modalities) linked to a document.

        Args:
            doc_id: The parent document identifier whose chunks should be
                deleted.
        """
        ...

    # ── Introspection ────────────────────────────────────────────────────

    @abstractmethod
    def health_check(self) -> dict:
        """Return a status snapshot of the store.

        The returned dict must include at minimum the count of stored items
        for each modality under the keys ``"images"``, ``"videos"``, and
        ``"audio"``.  Implementations may add additional diagnostic fields.

        Returns:
            A dict of the form::

                {
                    "images": <int>,
                    "videos": <int>,
                    "audio":  <int>,
                }

            with optional extra keys (e.g. ``"backend"``, ``"status"``).
        """
        ...
