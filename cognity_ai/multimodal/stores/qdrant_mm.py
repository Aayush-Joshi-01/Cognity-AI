"""
raglib.multimodal.stores.qdrant_mm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qdrant-backed multimodal vector store.

Uses three separate Qdrant collections — one per modality — each with its own
embedding dimensionality.  Payloads carry all chunk metadata so that
:class:`~raglib.multimodal.models.media.ImageChunk`,
:class:`~raglib.multimodal.models.media.VideoChunk`, and
:class:`~raglib.multimodal.models.media.AudioChunk` objects can be
reconstructed from query results without a second round-trip.

Filter dicts are translated to Qdrant ``Filter`` + ``FieldCondition`` objects;
see :meth:`QdrantMultimodalStore.query_images` for supported filter keys.

Requires::

    pip install qdrant-client

.. warning::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

from typing import Any, Optional

from cognity_ai.multimodal.models.media import (
    AudioChunk,
    ImageChunk,
    MultimodalRetrievalResult,
    VideoChunk,
)
from cognity_ai.multimodal.stores.base import BaseMultimodalStore


class QdrantMultimodalStore(BaseMultimodalStore):
    """Multimodal store backed by Qdrant with per-modality collections.

    Three named Qdrant collections are created on first use:

    - ``{collection_prefix}_images``   (dimension: *image_dims*)
    - ``{collection_prefix}_videos``   (dimension: *video_dims*)
    - ``{collection_prefix}_audio``    (dimension: *audio_dims*)

    All collections use cosine distance.

    Args:
        host: Qdrant server hostname (ignored when ``url`` is passed directly).
        port: Qdrant server port.
        api_key: Optional API key for Qdrant Cloud deployments.
        collection_prefix: Prefix for every collection name.
        image_dims: Embedding dimensionality for the image collection.
        video_dims: Embedding dimensionality for the video collection.
        audio_dims: Embedding dimensionality for the audio collection.

    Example::

        store = QdrantMultimodalStore(host="localhost", port=6333)
        store.upsert_image(image_chunk)
        results = store.query_images(query_embedding, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_prefix: str = "mm",
        image_dims: int = 512,
        video_dims: int = 512,
        audio_dims: int = 512,
    ) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._collection_prefix = collection_prefix
        self._image_dims = image_dims
        self._video_dims = video_dims
        self._audio_dims = audio_dims

        self._images_col = f"{collection_prefix}_images"
        self._videos_col = f"{collection_prefix}_videos"
        self._audio_col = f"{collection_prefix}_audio"

        self._client = None  # lazily initialised on first access

    # ── Lazy client / collection initialisation ──────────────────────────

    def _get_client(self):
        """Return (and lazily create) the Qdrant client with all collections."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required for QdrantMultimodalStore. "
                    "Install with: pip install qdrant-client"
                ) from exc

            self._client = QdrantClient(
                host=self._host,
                port=self._port,
                api_key=self._api_key or None,
            )

            _collection_dims = {
                self._images_col: self._image_dims,
                self._videos_col: self._video_dims,
                self._audio_col: self._audio_dims,
            }
            for cname, dims in _collection_dims.items():
                try:
                    self._client.get_collection(cname)
                except Exception:
                    self._client.create_collection(
                        collection_name=cname,
                        vectors_config=VectorParams(
                            size=dims,
                            distance=Distance.COSINE,
                        ),
                    )

        return self._client

    # ── Write operations ─────────────────────────────────────────────────

    def upsert_image(self, chunk: ImageChunk) -> None:
        """Store or update an image chunk.

        Image bytes are **not** stored in Qdrant; ``image_path`` is persisted
        as a payload field so the caller can load the image from disk when
        needed.

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

        from qdrant_client.models import PointStruct

        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id or "",
            "caption": chunk.caption or "",
            "ocr_text": chunk.ocr_text or "",
            "page_num": chunk.page_num if chunk.page_num is not None else -1,
            "source_path": chunk.source_path or "",
            "image_path": chunk.image_path or "",
        }
        payload.update(chunk.metadata)

        self._get_client().upsert(
            collection_name=self._images_col,
            points=[
                PointStruct(
                    id=_str_to_qdrant_id(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            ],
        )

    def upsert_video(self, chunk: VideoChunk) -> None:
        """Store or update a video chunk.

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

        from qdrant_client.models import PointStruct

        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "video_id": chunk.video_id,
            "title": chunk.title or "",
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "transcript": chunk.transcript or "",
            "scene_description": chunk.scene_description or "",
            "frame_count": len(chunk.frames),
            "scene_id": chunk.frames[0].scene_id or "" if chunk.frames else "",
        }
        payload.update(chunk.metadata)

        self._get_client().upsert(
            collection_name=self._videos_col,
            points=[
                PointStruct(
                    id=_str_to_qdrant_id(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            ],
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

        from qdrant_client.models import PointStruct

        duration_ms: float = 0.0
        if chunk.segments:
            duration_ms = chunk.segments[-1].end_ms - chunk.segments[0].start_ms

        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "audio_id": chunk.audio_id,
            "full_transcript": chunk.full_transcript or "",
            "duration_ms": duration_ms,
            "segment_count": len(chunk.segments),
            "source_path": chunk.source_path or "",
        }
        payload.update(chunk.metadata)

        self._get_client().upsert(
            collection_name=self._audio_col,
            points=[
                PointStruct(
                    id=_str_to_qdrant_id(chunk.id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            ],
        )

    # ── Query operations ─────────────────────────────────────────────────

    def query_images(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Query the image collection by embedding similarity.

        Supported filter keys:

        - ``"doc_id"``   — equality match on the ``doc_id`` payload field
        - ``"page_num"`` — equality match on the ``page_num`` payload field

        Args:
            embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Optional filter dict.

        Returns:
            List of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            with ``modality="image"`` and ``image_chunk`` populated.
        """
        qfilter = _build_qdrant_filter(filters)
        results = self._get_client().search(
            collection_name=self._images_col,
            query_vector=embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [_qdrant_hit_to_image_result(r) for r in results]

    def query_videos(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Query the video collection by embedding similarity.

        Supported filter keys:

        - ``"video_id"`` — equality match on the ``video_id`` payload field
        - ``"doc_id"``   — equality match on the ``doc_id`` payload field

        Args:
            embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Optional filter dict.

        Returns:
            List of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            with ``modality="video"`` and ``video_chunk`` populated.
        """
        qfilter = _build_qdrant_filter(filters)
        results = self._get_client().search(
            collection_name=self._videos_col,
            query_vector=embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [_qdrant_hit_to_video_result(r) for r in results]

    def query_audio(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[MultimodalRetrievalResult]:
        """Query the audio collection by embedding similarity.

        Supported filter keys:

        - ``"audio_id"`` — equality match on the ``audio_id`` payload field
        - ``"doc_id"``   — equality match on the ``doc_id`` payload field

        Args:
            embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Optional filter dict.

        Returns:
            List of :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
            with ``modality="audio"`` and ``audio_chunk`` populated.
        """
        qfilter = _build_qdrant_filter(filters)
        results = self._get_client().search(
            collection_name=self._audio_col,
            query_vector=embedding,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [_qdrant_hit_to_audio_result(r) for r in results]

    # ── Point-retrieval by ID ────────────────────────────────────────────

    def get_image_by_id(self, chunk_id: str) -> Optional[ImageChunk]:
        """Fetch an image chunk by its unique ID.

        Args:
            chunk_id: The chunk's unique identifier.

        Returns:
            :class:`~raglib.multimodal.models.media.ImageChunk` if found,
            otherwise ``None``.
        """
        points = self._get_client().retrieve(
            collection_name=self._images_col,
            ids=[_str_to_qdrant_id(chunk_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        p = points[0]
        meta = p.payload or {}
        return ImageChunk(
            id=chunk_id,
            doc_id=meta.get("doc_id") or None,
            image_path=meta.get("image_path") or None,
            caption=meta.get("caption") or None,
            ocr_text=meta.get("ocr_text") or None,
            page_num=meta.get("page_num") if meta.get("page_num", -1) >= 0 else None,
            source_path=meta.get("source_path") or None,
            embedding=list(p.vector) if p.vector is not None else None,
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
        points = self._get_client().retrieve(
            collection_name=self._videos_col,
            ids=[_str_to_qdrant_id(chunk_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        p = points[0]
        meta = p.payload or {}
        return VideoChunk(
            id=chunk_id,
            video_id=meta.get("video_id", ""),
            title=meta.get("title") or None,
            start_ms=float(meta.get("start_ms", 0)),
            end_ms=float(meta.get("end_ms", 0)),
            transcript=meta.get("transcript") or None,
            scene_description=meta.get("scene_description") or None,
            embedding=list(p.vector) if p.vector is not None else None,
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
        points = self._get_client().retrieve(
            collection_name=self._audio_col,
            ids=[_str_to_qdrant_id(chunk_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        p = points[0]
        meta = p.payload or {}
        return AudioChunk(
            id=chunk_id,
            audio_id=meta.get("audio_id", ""),
            full_transcript=meta.get("full_transcript") or None,
            source_path=meta.get("source_path") or None,
            embedding=list(p.vector) if p.vector is not None else None,
            metadata={k: v for k, v in meta.items() if k not in _AUDIO_RESERVED_KEYS},
        )

    # ── Deletion ─────────────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks across all modalities linked to *doc_id*.

        Deletes from all three collections any points whose ``doc_id`` payload
        field matches.  Video and audio chunks that originated from a document
        should include ``doc_id`` in their metadata when upserting.

        Args:
            doc_id: Parent document identifier.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        condition = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )
        client = self._get_client()
        for cname in [self._images_col, self._videos_col, self._audio_col]:
            try:
                client.delete(collection_name=cname, points_selector=condition)
            except Exception:
                pass  # collection may be empty / not yet created

    # ── Introspection ────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return item counts per modality plus connection info.

        Returns:
            Dict with keys ``"images"``, ``"videos"``, ``"audio"`` (counts),
            plus ``"backend"``, ``"host"``, ``"port"``.
        """
        client = self._get_client()

        def _count(cname: str) -> int:
            try:
                return client.get_collection(cname).points_count or 0
            except Exception:
                return 0

        return {
            "backend": "qdrant",
            "host": self._host,
            "port": self._port,
            "collection_prefix": self._collection_prefix,
            "images": _count(self._images_col),
            "videos": _count(self._videos_col),
            "audio": _count(self._audio_col),
        }


# ── Reserved payload key sets ─────────────────────────────────────────────────

_IMAGE_RESERVED_KEYS = frozenset(
    {
        "chunk_id",
        "doc_id",
        "caption",
        "ocr_text",
        "page_num",
        "source_path",
        "image_path",
    }
)
_VIDEO_RESERVED_KEYS = frozenset(
    {
        "chunk_id",
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
    {
        "chunk_id",
        "audio_id",
        "full_transcript",
        "duration_ms",
        "segment_count",
        "source_path",
    }
)


# ── Module-level helpers ──────────────────────────────────────────────────────


def _str_to_qdrant_id(value: str) -> int:
    """Convert a string ID to a positive 64-bit integer for Qdrant.

    Qdrant supports both string UUIDs and unsigned integer IDs.  We use the
    absolute hash so that the same string always maps to the same integer
    within a session.  Collisions are theoretically possible but extremely
    unlikely in practice.

    Args:
        value: Arbitrary string identifier.

    Returns:
        A non-negative 64-bit integer derived from the hash of *value*.
    """
    return abs(hash(value)) % (2**63)


def _build_qdrant_filter(filters: Optional[dict]):
    """Translate a simple filter dict to a Qdrant :class:`Filter`.

    Supports equality conditions on any payload key present in *filters*.
    All conditions are combined with ``must`` (logical AND).

    Args:
        filters: User-supplied filter dict, or ``None``.

    Returns:
        A Qdrant ``Filter`` object if any conditions apply, otherwise ``None``.
    """
    if not filters:
        return None

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ImportError:
        return None

    must_clauses = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    if not must_clauses:
        return None
    return Filter(must=must_clauses)


def _qdrant_hit_to_image_result(hit) -> MultimodalRetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` to a MultimodalRetrievalResult (image).

    Args:
        hit: A ``qdrant_client.models.ScoredPoint`` from a search call.

    Returns:
        :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
        with ``modality="image"``.
    """
    meta = hit.payload or {}
    chunk_id = meta.get("chunk_id", str(hit.id))
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
        score=hit.score,
        modality="image",
        text=meta.get("caption") or meta.get("ocr_text") or None,
        image_chunk=image_chunk,
        metadata=meta,
        source=meta.get("source_path") or meta.get("image_path") or None,
    )


def _qdrant_hit_to_video_result(hit) -> MultimodalRetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` to a MultimodalRetrievalResult (video).

    Args:
        hit: A ``qdrant_client.models.ScoredPoint`` from a search call.

    Returns:
        :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
        with ``modality="video"``.
    """
    meta = hit.payload or {}
    chunk_id = meta.get("chunk_id", str(hit.id))
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
        score=hit.score,
        modality="video",
        text=meta.get("transcript") or meta.get("scene_description") or None,
        video_chunk=video_chunk,
        metadata=meta,
        source=meta.get("video_id") or None,
    )


def _qdrant_hit_to_audio_result(hit) -> MultimodalRetrievalResult:
    """Convert a Qdrant ``ScoredPoint`` to a MultimodalRetrievalResult (audio).

    Args:
        hit: A ``qdrant_client.models.ScoredPoint`` from a search call.

    Returns:
        :class:`~raglib.multimodal.models.media.MultimodalRetrievalResult`
        with ``modality="audio"``.
    """
    meta = hit.payload or {}
    chunk_id = meta.get("chunk_id", str(hit.id))
    audio_chunk = AudioChunk(
        id=chunk_id,
        audio_id=meta.get("audio_id", ""),
        full_transcript=meta.get("full_transcript") or None,
        source_path=meta.get("source_path") or None,
        metadata={k: v for k, v in meta.items() if k not in _AUDIO_RESERVED_KEYS},
    )
    return MultimodalRetrievalResult(
        chunk_id=chunk_id,
        score=hit.score,
        modality="audio",
        text=meta.get("full_transcript") or None,
        audio_chunk=audio_chunk,
        metadata=meta,
        source=meta.get("source_path") or meta.get("audio_id") or None,
    )
