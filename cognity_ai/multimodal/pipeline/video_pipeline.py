"""
raglib.multimodal.pipeline.video_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video ingestion pipeline for multimodal RAG.

Flow::

  Video file → VideoLoader (frames + audio track)
             → Frame embedding via multimodal embedder (CLIP/SigLIP)
             → Audio extraction → Transcription
             → Temporal chunking (by scene or fixed duration)
             → VideoChunk embedding (mean pool of frame embeddings)
             → Store frames + chunks in multimodal store

The pipeline delegates frame loading, embedding, and transcription to their
respective subsystems, acting purely as an orchestrator.

.. warning::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.

Example::

    from cognity_ai.multimodal.embedders.clip import CLIPEmbedder
    from cognity_ai.multimodal.stores.chroma_mm import ChromaMultimodalStore
    from cognity_ai.multimodal.pipeline.video_pipeline import VideoIngestionPipeline

    embedder = CLIPEmbedder()
    store = ChromaMultimodalStore()
    pipeline = VideoIngestionPipeline(embedder=embedder, store=store)

    chunk = pipeline.ingest("lecture.mp4", doc_id="doc-lecture-01")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from cognity_ai.multimodal.models.media import VideoChunk, VideoFrame
from cognity_ai.multimodal.stores.base import BaseMultimodalStore

logger = logging.getLogger(__name__)


class VideoIngestionPipeline:
    """End-to-end pipeline for ingesting video files into a multimodal store.

    The pipeline performs the following steps for each video:

    1. Load video frames (and optionally the audio track) via a
       :class:`VideoLoader` instance.
    2. Embed each frame using a multimodal embedder's
       ``embed_image(image_bytes_b64)`` method.
    3. Optionally transcribe the audio track with a *transcriber*.
    4. Align transcript segments to frames by timestamp.
    5. Compute a chunk-level embedding as the mean pool of all frame
       embeddings.
    6. Construct a :class:`~raglib.multimodal.models.media.VideoChunk` with
       the embedded frames, merged transcript, and scene description.
    7. Upsert the chunk into the provided store.
    8. Return the chunk.

    Args:
        embedder: A
            :class:`~raglib.multimodal.embedders.base.BaseMultimodalEmbedder`
            instance whose ``embed_image`` method is used for frame embedding.
        store: A
            :class:`~raglib.multimodal.stores.base.BaseMultimodalStore`
            instance into which ingested chunks are upserted.
        transcriber: Optional transcriber that implements
            ``transcribe(audio_path: str) -> TranscriptionResult``.  When
            supplied, the audio track is extracted and transcribed; each
            :class:`~raglib.multimodal.models.media.VideoFrame`'s transcript
            segment is aligned by timestamp.
        loader: Optional :class:`VideoLoader` instance.  A default loader is
            created from the file path if ``None`` is passed.
    """

    def __init__(
        self,
        embedder,
        store: BaseMultimodalStore,
        transcriber=None,
        loader=None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._transcriber = transcriber
        self._loader = loader

    # ── Public API ────────────────────────────────────────────────────────

    def ingest(
        self,
        video_path: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> VideoChunk:
        """Ingest a single video file.

        Args:
            video_path: Absolute or relative path to the video file.
                Supported formats depend on the underlying :class:`VideoLoader`
                and the OpenCV / av installation (MP4, MKV, AVI, MOV, …).
            doc_id: Optional parent document identifier.
            metadata: Arbitrary key-value pairs merged into the
                :attr:`~raglib.multimodal.models.media.VideoChunk.metadata`
                field before storing.

        Returns:
            The ingested :class:`~raglib.multimodal.models.media.VideoChunk`
            with ``embedding`` populated.

        Raises:
            FileNotFoundError: If *video_path* does not exist on disk.
            ValueError: If no frames could be loaded from the video.
        """
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path!r}")

        logger.debug("Ingesting video: %s", path)

        # Step 1 — load video frames
        loader = self._loader or _DefaultVideoLoader()
        raw_frames = loader.load(str(path))  # list of _RawFrame(bytes_b64, ts_ms, index)
        if not raw_frames:
            raise ValueError(f"No frames could be extracted from video: {video_path!r}")

        video_id = _sha256_id(str(path))

        # Step 2 — embed each frame
        frames: list[VideoFrame] = []
        frame_embeddings: list[list[float]] = []
        for raw in raw_frames:
            try:
                emb = self._embedder.embed_image(raw.image_bytes_b64)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Frame embedding failed at ts=%.0fms: %s", raw.timestamp_ms, exc
                )
                emb = None

            frame = VideoFrame(
                frame_id=f"{video_id}_{raw.frame_index}",
                video_id=video_id,
                frame_index=raw.frame_index,
                timestamp_ms=raw.timestamp_ms,
                image_bytes_b64=raw.image_bytes_b64,
                embedding=emb,
                scene_id=raw.scene_id,
            )
            frames.append(frame)
            if emb:
                frame_embeddings.append(emb)

        if not frame_embeddings:
            raise ValueError(
                f"Embedder produced no embeddings for any frame in: {video_path!r}"
            )

        # Step 3 — optional transcription
        transcript: Optional[str] = None
        transcript_segments: list = []
        if self._transcriber is not None:
            try:
                result = self._transcriber.transcribe(str(path))
                transcript = result.full_text if hasattr(result, "full_text") else str(result)
                # Supports both TranscriptionResult (TimestampedSegment with
                # start_ms/end_ms) and plain dicts with start/end in seconds.
                transcript_segments = (
                    result.segments if hasattr(result, "segments") else []
                )
                logger.debug(
                    "Transcription produced %d chars for %s",
                    len(transcript or ""),
                    path.name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transcription failed for %s: %s", path.name, exc)

        # Step 4 — align transcript segments to frames by timestamp
        if transcript_segments:
            frames = _align_transcript_to_frames(frames, transcript_segments)

        # Step 5 — compute chunk embedding as mean pool of frame embeddings
        chunk_embedding = _mean_pool(frame_embeddings)

        # Step 6 — build chunk
        start_ms = frames[0].timestamp_ms if frames else 0.0
        end_ms = frames[-1].timestamp_ms if frames else 0.0
        chunk = VideoChunk(
            id=video_id,
            video_id=video_id,
            title=path.stem,
            frames=frames,
            start_ms=start_ms,
            end_ms=end_ms,
            transcript=transcript,
            embedding=chunk_embedding,
            scene_description=None,  # can be populated by a VLM in a later pass
            metadata={**(metadata or {}), "source_path": str(path), "doc_id": doc_id or ""},
        )

        # Step 7 — store
        self._store.upsert_video(chunk)
        logger.info(
            "Ingested video chunk %s from %s (%d frames)",
            video_id,
            path.name,
            len(frames),
        )

        return chunk

    def ingest_batch(
        self,
        video_paths: list[str],
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[VideoChunk]:
        """Ingest multiple video files sequentially.

        Args:
            video_paths: List of paths to video files.
            doc_id: Optional shared parent document identifier.
            metadata: Optional shared metadata applied to all videos.

        Returns:
            List of ingested
            :class:`~raglib.multimodal.models.media.VideoChunk` objects.
            Failures are logged and skipped.
        """
        chunks: list[VideoChunk] = []
        for path in video_paths:
            try:
                chunk = self.ingest(path, doc_id=doc_id, metadata=metadata)
                chunks.append(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest video %r: %s", path, exc)
        return chunks


# ── Internal video loader ─────────────────────────────────────────────────────


class _RawFrame:
    """Lightweight container for a decoded video frame."""

    __slots__ = ("image_bytes_b64", "timestamp_ms", "frame_index", "scene_id")

    def __init__(
        self,
        image_bytes_b64: str,
        timestamp_ms: float,
        frame_index: int,
        scene_id: Optional[str] = None,
    ) -> None:
        self.image_bytes_b64 = image_bytes_b64
        self.timestamp_ms = timestamp_ms
        self.frame_index = frame_index
        self.scene_id = scene_id


class _DefaultVideoLoader:
    """Minimal video loader using OpenCV.

    Extracts one frame per second (or every *sample_rate_ms* milliseconds) to
    keep the number of embeddings manageable.  Callers may replace this loader
    with a scene-boundary-aware implementation for higher fidelity.

    Requires::

        pip install opencv-python
    """

    def __init__(self, sample_rate_ms: float = 1000.0) -> None:
        """Initialise the loader.

        Args:
            sample_rate_ms: Interval between sampled frames in milliseconds.
                Default is 1 000 ms (one frame per second).
        """
        self._sample_rate_ms = sample_rate_ms

    def load(self, video_path: str) -> list[_RawFrame]:
        """Extract frames from *video_path* at the configured sample rate.

        Args:
            video_path: Path to the video file.

        Returns:
            List of :class:`_RawFrame` objects in temporal order.

        Raises:
            ImportError: If OpenCV is not installed.
            IOError: If the video file cannot be opened.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for the default VideoLoader. "
                "Install with: pip install opencv-python"
            ) from exc

        import base64

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path!r}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_every_n = max(1, int(round((self._sample_rate_ms / 1000.0) * fps)))

        frames: list[_RawFrame] = []
        frame_index = 0
        read_index = 0

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if read_index % sample_every_n == 0:
                # Convert BGR → JPEG bytes → base-64
                _, jpeg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(jpeg_buf.tobytes()).decode("utf-8")
                ts_ms = (read_index / fps) * 1000.0
                frames.append(_RawFrame(b64, ts_ms, frame_index))
                frame_index += 1
            read_index += 1

        cap.release()
        logger.debug(
            "VideoLoader extracted %d frames from %s", len(frames), video_path
        )
        return frames


# ── Module-level helpers ──────────────────────────────────────────────────────


def _mean_pool(embeddings: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of embedding vectors.

    Args:
        embeddings: Non-empty list of same-length float vectors.

    Returns:
        A single vector of the same length as the inputs.
    """
    if not embeddings:
        return []
    dims = len(embeddings[0])
    total = [0.0] * dims
    for emb in embeddings:
        for i, v in enumerate(emb):
            total[i] += v
    n = len(embeddings)
    return [v / n for v in total]


def _align_transcript_to_frames(
    frames: list[VideoFrame],
    segments: list,
) -> list[VideoFrame]:
    """Attach transcript text to each frame based on timestamp overlap.

    For each frame, the function finds all transcript segments whose time
    range overlaps with the frame's timestamp and stores the concatenated text
    in ``frame.metadata["transcript_segment"]``.

    Supports two segment formats:

    - :class:`~raglib.multimodal.transcribers.base.TimestampedSegment` Pydantic
      models (``start_ms`` / ``end_ms`` in milliseconds, ``text`` attribute).
    - Plain ``dict`` objects with ``"start"`` / ``"end"`` keys in **seconds**
      and a ``"text"`` key (compatible with Whisper JSON output).

    Args:
        frames: List of :class:`~raglib.multimodal.models.media.VideoFrame`
            objects with ``timestamp_ms`` populated.
        segments: List of transcript segments from an ASR result.

    Returns:
        The same list of frames, each with an updated ``metadata`` dict.
    """
    for frame in frames:
        ts_ms = frame.timestamp_ms
        overlapping: list[str] = []
        for seg in segments:
            if hasattr(seg, "start_ms"):
                # TimestampedSegment (raglib.multimodal.transcribers.base)
                if seg.start_ms <= ts_ms <= seg.end_ms:
                    overlapping.append(seg.text)
            else:
                # Plain dict with start/end in seconds
                start_ms = seg.get("start", 0) * 1000.0
                end_ms = seg.get("end", float("inf")) * 1000.0
                if start_ms <= ts_ms <= end_ms:
                    overlapping.append(seg.get("text", ""))
        if overlapping:
            frame.metadata["transcript_segment"] = " ".join(overlapping).strip()
    return frames


def _sha256_id(value: str) -> str:
    """Return the first 16 hex chars of the SHA-256 digest of *value*.

    Args:
        value: Input string (typically a resolved file path).

    Returns:
        16-character hex string suitable for use as a chunk ID.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
