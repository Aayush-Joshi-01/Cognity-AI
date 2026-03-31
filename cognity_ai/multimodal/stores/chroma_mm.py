"""
raglib.multimodal.stores.chroma_mm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ChromaDB-backed multimodal vector store.

Uses three separate Chroma collections — one per modality — so that each
modality can have its own embedding dimensionality and metadata schema.
Image bytes are intentionally *not* stored inside ChromaDB (blobs would
inflate the database unreasonably); instead, ``image_path`` is recorded in
the collection metadata so callers can load the image from disk on demand.

Requires::

    pip install chromadb

.. warning::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from cognity_ai.multimodal.models.media import (
    AudioChunk,
    AudioSegment,
    ImageChunk,
    MultimodalRetrievalResult,
    VideoChunk,
    VideoFrame,
)
from cognity_ai.multimodal.stores.base import BaseMultimodalStore


class ChromaMultimodalStore(BaseMultimodalStore):
    """Multimodal store backed by ChromaDB with per-modality collections.

    Three named collections are created (or opened if they already exist):

    - ``{collection_prefix}_images``
    - ``{collection_prefix}_videos``
    - ``{collection_prefix}_audio``

    All collections use cosine distance for similarity search.

    Args:
        persist_directory: Directory where ChromaDB will persist data on disk.
        collection_prefix: Prefix prepended to every collection name, allowing
            multiple independent store instances to share one ChromaDB
            deployment.

    Example::

        store = ChromaMultimodalStore(persist_directory="./chroma_mm")
        store.upsert_image(image_chunk)
        results = store.query_images(query_embedding, top_k=5)
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_multimodal",
        collection_prefix: str = "mm",
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_prefix = collection_prefix
        self._client = None  # lazily initialised on first access

    # ── Lazy client / collection initialisation ──────────────────────────

    def _get_client(self):
        """Return (and lazily create) the ChromaDB persistent client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                raise ImportError(
                    "ChromaDB is required for ChromaMultimodalStore. "
                    "Install with: pip install chromadb"
                ) from exc

            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._images = self._client.get_or_create_collection(
                name=f"{self._collection_prefix}_images",
                metadata={"hnsw:space": "cosine"},
            )
            self._videos = self._client.get_or_create_collection(
                name=f"{self._collection_prefix}_videos",
                metadata={"hnsw:space": "cosine"},
            )
            self._audio = self._client.get_or_create_collection(
                name=f"{self._collection_prefix}_audio",
                metadata={"hnsw:space": "cosine"},
            )
        return self._client

    @property
    def _images_col(self):
        self._get_client()
        return self._images

    @property
    def _videos_col(self):
        self._get_client()
        return self._videos

    @property
    def _audio_col(self):
        self._get_client()
        return self._audio

    # ── Write operations ─────────────────────────────────────────────────

    def upsert_image(self, chunk: ImageChunk) -> None:
        """Store or update an image chunk.

        Image bytes are **not** stored in ChromaDB. The ``image_path`` field
        of the chunk is recorded as metadata instead.

        Args:
            chunk: :class:`~raglib.multimodal.models.media.ImageChunk` with a
                populated ``embedding``.

        Raises:
            ValueError: If ``chunk.embedding`` is ``None``.
        """
        if chunk.embedding is None:
            raise ValueError(
                f"ImageChunk {chunk.id!r} has no embedding; run the embedder first."
            )

        metadata: dict[str, Any] = {
            "doc_id": chunk.doc_id or "",
            "caption": chunk.caption or "",
            "ocr_text": chunk.ocr_text or "",
            "page_num": chunk.page_num if chunk.page_num is not None else -1,
            "source_path": chunk.source_path or "",
            "image_path": chunk.image_path or "",
        }
        # Merge any extra user-supplied metadata (must be ChromaDB-safe types)
        metadata.update(_flatten_metadata(chunk.metadata))

        self._images_col.upsert(
            ids=[chunk.id],
            embeddings=[chunk.embedding],
            documents=[chunk.caption or chunk.ocr_text or ""],
            metadatas=[metadata],
        )

    def upsert_video(self, chunk: VideoChunk) -> None:
        """Store or update a video chunk.

        Frame-level embeddings are embedded in a JSON-serialised ``frames``
        metadata field (lightweight summary — not full bytes).

        Args:
            chunk: :class:`~raglib.multimodal.models.media.VideoChunk` with a
                populated ``embedding``.

        Raises:
            ValueError: If ``chunk.embedding`` is ``None``.
        """
        if chunk.embedding is None:
            raise ValueError(
                f"VideoChunk {chunk.id!r} has no embedding; run the embedder first."
            )

        metadata: dict[str, Any] = {
            "video_id": chunk.video_id,
            "title": chunk.title or "",
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "transcript": chunk.transcript or "",
            "scene_description": chunk.scene_description or "",
            "frame_count": len(chunk.frames),
            # Store the scene_id of the first frame if present (representative)
            "scene_id": chunk.frames[0].scene_id or "" if chunk.frames else "",
        }
        metadata.update(_flatten_metadata(chunk.metadata))

        self._videos_col.upsert(
            ids=[chunk.id],
            embeddings=[chunk.embedding],
            documents=[chunk.transcript or chunk.scene_description or ""],
            metadatas=[metadata],
        )

    def upsert_audio(self, chunk: AudioChunk) -> None:
        """Store or update an audio chunk.

        Args:
            chunk: :class:`~raglib.multimodal.models.media.AudioChunk` with a
                populated ``embedding``.

        Raises:
            ValueError: If ``chunk.embedding`` is ``None``.
        """
        if chunk.embedding is None:
            raise ValueError(
                f"AudioChunk {chunk.id!r} has no embedding; run the embedder first."
            )

        # Compute duration from segments when available
        duration_ms: float = 0.0
        if chunk.segments:
            duration_ms = chunk.segments[-1].end_ms - chunk.segments[0].start_ms

        metadata: dict[str, Any] = {
            "audio_id": chunk.audio_id,
            "full_transcript": chunk.full_transcript or "",
            "duration_ms": duration_ms,
            "segment_count": len(chunk.segments),
            "source_path": chunk.source_path or "",
        }
        metadata.update(_flatten_metadata(chunk.metadata))

        self._audio_col.upsert(
            ids=[chunk.id],
            embeddings=[chunk.embedding],
            documents=[chunk.full_transcript or ""],
            metadatas=[metadata],
        )

    # ── Query operations ─────────────────────────────────────────────────

    def query_images(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Query image collection by embedding similarity.

        Args:
            embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Supported keys: ``doc_id``, ``page_num``.

        Returns:
            List of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            with ``modality="image"`` and ``image_chunk`` populated.
        """
        where = _build_chroma_where(filters)
        count = self._images_col.count()
        if count == 0:
            return []
        n = min(top_k, count)

        results = self._images_col.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        return [
            _chroma_row_to_image_result(cid, doc, meta, dist)
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def query_videos(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Query video collection by embedding similarity.

        Args:
            embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Supported keys: ``video_id``.

        Returns:
            List of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            with ``modality="video"`` and ``video_chunk`` populated.
        """
        where = _build_chroma_where(filters)
        count = self._videos_col.count()
        if count == 0:
            return []
        n = min(top_k, count)

        results = self._videos_col.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        return [
            _chroma_row_to_video_result(cid, doc, meta, dist)
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def query_audio(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Query audio collection by embedding similarity.

        Args:
            embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Supported keys: ``audio_id``.

        Returns:
            List of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            with ``modality="audio"`` and ``audio_chunk`` populated.
        """
        where = _build_chroma_where(filters)
        count = self._audio_col.count()
        if count == 0:
            return []
        n = min(top_k, count)

        results = self._audio_col.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        return [
            _chroma_row_to_audio_result(cid, doc, meta, dist)
            for cid, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ── Point-retrieval by ID ────────────────────────────────────────────

    def get_image_by_id(self, chunk_id: str) -> Optional[ImageChunk]:
        """Fetch an image chunk by its unique ID.

        Args:
            chunk_id: The chunk's unique identifier.

        Returns:
            :class:`~raglib.multimodal.models.media.ImageChunk` if found,
            otherwise ``None``.
        """
        result = self._images_col.get(
            ids=[chunk_id],
            include=["metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        embedding = result["embeddings"][0] if result.get("embeddings") else None
        return ImageChunk(
            id=chunk_id,
            doc_id=meta.get("doc_id") or None,
            image_path=meta.get("image_path") or None,
            caption=meta.get("caption") or None,
            ocr_text=meta.get("ocr_text") or None,
            page_num=meta.get("page_num") if meta.get("page_num", -1) >= 0 else None,
            source_path=meta.get("source_path") or None,
            embedding=list(embedding) if embedding is not None else None,
            metadata={k: v for k, v in meta.items() if k not in _IMAGE_RESERVED_KEYS},
        )

    def get_video_by_id(self, chunk_id: str) -> Optional[VideoChunk]:
        """Fetch a video chunk by its unique ID.

        Args:
            chunk_id: The chunk's unique identifier.

        Returns:
            :class:`~raglib.multimodal.models.media.VideoChunk` if found,
            otherwise ``None``.
        """
        result = self._videos_col.get(
            ids=[chunk_id],
            include=["metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        embedding = result["embeddings"][0] if result.get("embeddings") else None
        return VideoChunk(
            id=chunk_id,
            video_id=meta.get("video_id", ""),
            title=meta.get("title") or None,
            start_ms=float(meta.get("start_ms", 0)),
            end_ms=float(meta.get("end_ms", 0)),
            transcript=meta.get("transcript") or None,
            scene_description=meta.get("scene_description") or None,
            embedding=list(embedding) if embedding is not None else None,
            metadata={k: v for k, v in meta.items() if k not in _VIDEO_RESERVED_KEYS},
        )

    def get_audio_by_id(self, chunk_id: str) -> Optional[AudioChunk]:
        """Fetch an audio chunk by its unique ID.

        Args:
            chunk_id: The chunk's unique identifier.

        Returns:
            :class:`~raglib.multimodal.models.media.AudioChunk` if found,
            otherwise ``None``.
        """
        result = self._audio_col.get(
            ids=[chunk_id],
            include=["metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        embedding = result["embeddings"][0] if result.get("embeddings") else None
        return AudioChunk(
            id=chunk_id,
            audio_id=meta.get("audio_id", ""),
            full_transcript=meta.get("full_transcript") or None,
            source_path=meta.get("source_path") or None,
            embedding=list(embedding) if embedding is not None else None,
            metadata={k: v for k, v in meta.items() if k not in _AUDIO_RESERVED_KEYS},
        )

    # ── Deletion ─────────────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks across all modalities linked to *doc_id*.

        Args:
            doc_id: Parent document identifier whose chunks should be removed.
        """
        _chroma_delete_where(self._images_col, {"doc_id": doc_id})
        # Videos and audio use ``video_id`` / ``audio_id`` as their primary
        # key, but user-supplied metadata may include a ``doc_id`` field.
        _chroma_delete_where(self._videos_col, {"doc_id": doc_id})
        _chroma_delete_where(self._audio_col, {"doc_id": doc_id})

    # ── Introspection ────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return item counts per modality.

        Returns:
            Dict with keys ``"images"``, ``"videos"``, ``"audio"`` mapping to
            integer counts, plus ``"backend"`` and ``"persist_directory"``.
        """
        return {
            "backend": "chromadb",
            "persist_directory": self._persist_directory,
            "collection_prefix": self._collection_prefix,
            "images": self._images_col.count(),
            "videos": self._videos_col.count(),
            "audio": self._audio_col.count(),
        }


# ── Reserved metadata key sets (used to separate extra metadata on read-back) ─

_IMAGE_RESERVED_KEYS = frozenset(
    {"doc_id", "caption", "ocr_text", "page_num", "source_path", "image_path"}
)
_VIDEO_RESERVED_KEYS = frozenset(
    {
        "video_id",
        "title",
        "start_ms",
        "end_ms",
        "transcript",
        "scene_description",
        "frame_count",
        "scene_id",
    }
)
_AUDIO_RESERVED_KEYS = frozenset(
    {"audio_id", "full_transcript", "duration_ms", "segment_count", "source_path"}
)


# ── Module-level helpers ──────────────────────────────────────────────────────


def _flatten_metadata(meta: dict) -> dict:
    """Flatten a metadata dict to ChromaDB-safe scalar types.

    ChromaDB only supports ``str``, ``int``, ``float``, and ``bool`` values in
    metadata.  Any complex value is JSON-serialised into a string.

    Args:
        meta: Arbitrary metadata dict from a chunk model.

    Returns:
        A new dict with only ChromaDB-safe value types.
    """
    flat: dict = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            try:
                flat[key] = json.dumps(value)
            except (TypeError, ValueError):
                flat[key] = str(value)
    return flat


def _build_chroma_where(filters: Optional[dict]) -> Optional[dict]:
    """Translate a simple filter dict to a ChromaDB ``where`` clause.

    Supported filter keys and their ChromaDB translations:

    - ``"doc_id"``   → ``{"doc_id": {"$eq": value}}``
    - ``"page_num"`` → ``{"page_num": {"$eq": value}}``
    - ``"video_id"`` → ``{"video_id": {"$eq": value}}``
    - ``"audio_id"`` → ``{"audio_id": {"$eq": value}}``

    Multiple conditions are combined with ``$and``.

    Args:
        filters: User-supplied filter dict, or ``None``.

    Returns:
        ChromaDB-compatible ``where`` dict, or ``None`` if no filters apply.
    """
    if not filters:
        return None

    _supported = {"doc_id", "page_num", "video_id", "audio_id"}
    clauses = [
        {k: {"$eq": v}}
        for k, v in filters.items()
        if k in _supported
    ]

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _chroma_delete_where(collection, filters: dict) -> None:
    """Delete all points from *collection* matching *filters*.

    Falls back to a ``get`` + ``delete by ids`` pattern if the ``delete``
    call raises (some versions of ChromaDB do not support ``where`` in
    ``delete``).

    Args:
        collection: A ChromaDB collection object.
        filters: Simple equality filter dict (``{field: value}``).
    """
    where = _build_chroma_where(filters)
    if where is None:
        return
    try:
        collection.delete(where=where)
    except Exception:
        result = collection.get(where=where)
        if result["ids"]:
            collection.delete(ids=result["ids"])


def _chroma_row_to_image_result(
    chunk_id: str, document: str, meta: dict, distance: float
) -> MultimodalRetrievalResult:
    """Convert a single ChromaDB query row to a MultimodalRetrievalResult (image).

    Args:
        chunk_id: ChromaDB document ID.
        document: The stored document string (caption / OCR text).
        meta: Metadata dict from ChromaDB.
        distance: Cosine distance (0 = identical, 2 = opposite).

    Returns:
        A :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
        with ``modality="image"``.
    """
    score = 1.0 - distance  # cosine distance → similarity
    image_chunk = ImageChunk(
        id=chunk_id,
        doc_id=meta.get("doc_id") or None,
        image_path=meta.get("image_path") or None,
        caption=meta.get("caption") or None,
        ocr_text=meta.get("ocr_text") or None,
        page_num=meta.get("page_num") if meta.get("page_num", -1) >= 0 else None,
        source_path=meta.get("source_path") or None,
        metadata={k: v for k, v in meta.items() if k not in _IMAGE_RESERVED_KEYS},
    )
    return MultimodalRetrievalResult(
        chunk_id=chunk_id,
        score=score,
        modality="image",
        text=document or None,
        image_chunk=image_chunk,
        metadata=meta,
        source=meta.get("source_path") or meta.get("image_path") or None,
    )


def _chroma_row_to_video_result(
    chunk_id: str, document: str, meta: dict, distance: float
) -> MultimodalRetrievalResult:
    """Convert a single ChromaDB query row to a MultimodalRetrievalResult (video).

    Args:
        chunk_id: ChromaDB document ID.
        document: The stored document string (transcript / scene description).
        meta: Metadata dict from ChromaDB.
        distance: Cosine distance.

    Returns:
        A :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
        with ``modality="video"``.
    """
    score = 1.0 - distance
    video_chunk = VideoChunk(
        id=chunk_id,
        video_id=meta.get("video_id", ""),
        title=meta.get("title") or None,
        start_ms=float(meta.get("start_ms", 0)),
        end_ms=float(meta.get("end_ms", 0)),
        transcript=meta.get("transcript") or None,
        scene_description=meta.get("scene_description") or None,
        metadata={k: v for k, v in meta.items() if k not in _VIDEO_RESERVED_KEYS},
    )
    return MultimodalRetrievalResult(
        chunk_id=chunk_id,
        score=score,
        modality="video",
        text=document or None,
        video_chunk=video_chunk,
        metadata=meta,
        source=meta.get("video_id") or None,
    )


def _chroma_row_to_audio_result(
    chunk_id: str, document: str, meta: dict, distance: float
) -> MultimodalRetrievalResult:
    """Convert a single ChromaDB query row to a MultimodalRetrievalResult (audio).

    Args:
        chunk_id: ChromaDB document ID.
        document: The stored document string (full transcript).
        meta: Metadata dict from ChromaDB.
        distance: Cosine distance.

    Returns:
        A :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
        with ``modality="audio"``.
    """
    score = 1.0 - distance
    audio_chunk = AudioChunk(
        id=chunk_id,
        audio_id=meta.get("audio_id", ""),
        full_transcript=meta.get("full_transcript") or None,
        source_path=meta.get("source_path") or None,
        metadata={k: v for k, v in meta.items() if k not in _AUDIO_RESERVED_KEYS},
    )
    return MultimodalRetrievalResult(
        chunk_id=chunk_id,
        score=score,
        modality="audio",
        text=document or None,
        audio_chunk=audio_chunk,
        metadata=meta,
        source=meta.get("source_path") or meta.get("audio_id") or None,
    )
