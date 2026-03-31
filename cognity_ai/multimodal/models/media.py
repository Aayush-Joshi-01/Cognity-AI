"""
raglib.multimodal.models.media
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pydantic v2 data models for multimodal RAG chunks and retrieval results.

Covers image, video (frame + chunk), audio (segment + chunk), and a unified
``MultimodalRetrievalResult`` that can carry any modality alongside a relevance
score.

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ImageChunk(BaseModel):
    """A retrievable chunk that represents a single image or image region.

    Attributes:
        id: Unique identifier for this chunk.
        doc_id: Identifier of the parent document this image belongs to.
        image_path: Filesystem or URL path to the image file, if available.
        image_bytes_b64: Base-64-encoded raw image bytes, if the image is
            stored inline rather than on disk.
        caption: Human-readable or auto-generated caption describing the image.
        ocr_text: OCR-extracted text found within the image, if any.
        embedding: Dense vector representation of the image produced by an
            embedder.  ``None`` until an embedding pass has been run.
        metadata: Arbitrary key-value metadata (e.g. EXIF data, loader hints).
        page_num: 1-indexed page number within a multi-page document (e.g. PDF)
            from which this image was extracted.
        source_path: Path to the originating source document.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique chunk identifier.")
    doc_id: Optional[str] = Field(
        default=None, description="Parent document identifier."
    )
    image_path: Optional[str] = Field(
        default=None, description="Filesystem or URL path to the image."
    )
    image_bytes_b64: Optional[str] = Field(
        default=None, description="Base-64-encoded image bytes (inline storage)."
    )
    caption: Optional[str] = Field(
        default=None, description="Caption or description for this image."
    )
    ocr_text: Optional[str] = Field(
        default=None, description="Text extracted from the image via OCR."
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Dense embedding vector for this image chunk."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )
    page_num: Optional[int] = Field(
        default=None, description="1-indexed page number within a multi-page document."
    )
    source_path: Optional[str] = Field(
        default=None, description="Path to the originating source document."
    )


class VideoFrame(BaseModel):
    """A single decoded frame extracted from a video.

    Attributes:
        frame_id: Unique identifier for this frame.
        video_id: Identifier of the parent video.
        frame_index: Zero-based index of the frame in the video stream.
        timestamp_ms: Position of the frame in the video, in milliseconds.
        image_bytes_b64: Base-64-encoded JPEG/PNG bytes for this frame.
        embedding: Dense vector representation of the frame image.
        scene_id: Optional identifier linking this frame to a detected scene
            boundary / scene cluster.
        metadata: Arbitrary key-value metadata (codec info, resolution, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: str = Field(..., description="Unique frame identifier.")
    video_id: str = Field(..., description="Parent video identifier.")
    frame_index: int = Field(..., description="Zero-based frame index.")
    timestamp_ms: float = Field(
        ..., description="Frame timestamp within the video, in milliseconds."
    )
    image_bytes_b64: Optional[str] = Field(
        default=None, description="Base-64-encoded image bytes for this frame."
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Dense embedding vector for this frame."
    )
    scene_id: Optional[str] = Field(
        default=None, description="Scene cluster identifier for this frame."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )


class VideoChunk(BaseModel):
    """A temporal segment of a video, composed of one or more frames.

    A ``VideoChunk`` is the unit of retrieval for video content.  It spans a
    contiguous time window and may carry a transcript of spoken content as well
    as an aggregate embedding representing the whole segment.

    Attributes:
        id: Unique identifier for this chunk.
        video_id: Identifier of the parent video.
        title: Optional human-readable title or label for the segment.
        frames: Ordered list of :class:`VideoFrame` objects that belong to
            this chunk.
        start_ms: Start timestamp of the chunk in milliseconds.
        end_ms: End timestamp of the chunk in milliseconds.
        transcript: Speech-to-text transcript of audio within this segment.
        embedding: Aggregate dense vector for the whole chunk (e.g. mean of
            frame embeddings, or a dedicated video encoder output).
        scene_description: Auto-generated or human-provided textual description
            of the visual scene.
        metadata: Arbitrary key-value metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique chunk identifier.")
    video_id: str = Field(..., description="Parent video identifier.")
    title: Optional[str] = Field(
        default=None, description="Human-readable title for this video segment."
    )
    frames: list[VideoFrame] = Field(
        default_factory=list, description="Frames belonging to this chunk."
    )
    start_ms: float = Field(
        ..., description="Chunk start timestamp in milliseconds."
    )
    end_ms: float = Field(
        ..., description="Chunk end timestamp in milliseconds."
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Speech-to-text transcript of spoken content in this chunk.",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Aggregate dense embedding vector for this video chunk.",
    )
    scene_description: Optional[str] = Field(
        default=None, description="Textual description of the visual scene."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )


class AudioSegment(BaseModel):
    """A single utterance or diarization segment within an audio stream.

    Attributes:
        id: Unique identifier for this segment.
        audio_id: Identifier of the parent audio file / chunk.
        start_ms: Segment start time in milliseconds.
        end_ms: Segment end time in milliseconds.
        transcript: Transcribed text for this segment.
        embedding: Dense vector representation of the audio segment.
        speaker_id: Diarization speaker label (e.g. ``"SPEAKER_00"``).
        confidence: ASR confidence score in the range ``[0.0, 1.0]``.
        metadata: Arbitrary key-value metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique segment identifier.")
    audio_id: str = Field(..., description="Parent audio file/chunk identifier.")
    start_ms: float = Field(..., description="Segment start time in milliseconds.")
    end_ms: float = Field(..., description="Segment end time in milliseconds.")
    transcript: Optional[str] = Field(
        default=None, description="Transcribed text for this segment."
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Dense embedding vector for this audio segment."
    )
    speaker_id: Optional[str] = Field(
        default=None, description="Diarization speaker label."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ASR transcription confidence score in [0, 1].",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )


class AudioChunk(BaseModel):
    """A retrievable chunk representing a contiguous section of audio.

    An ``AudioChunk`` groups one or more :class:`AudioSegment` objects (e.g.
    from speaker diarization) into a single retrieval unit, with a combined
    full transcript and an aggregate embedding.

    Attributes:
        id: Unique identifier for this chunk.
        audio_id: Identifier of the parent audio file.
        segments: Ordered list of :class:`AudioSegment` objects.
        full_transcript: Concatenated transcript of all segments.
        embedding: Aggregate dense vector for the whole chunk.
        metadata: Arbitrary key-value metadata.
        source_path: Path to the originating audio file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique chunk identifier.")
    audio_id: str = Field(..., description="Parent audio file identifier.")
    segments: list[AudioSegment] = Field(
        default_factory=list, description="Audio segments composing this chunk."
    )
    full_transcript: Optional[str] = Field(
        default=None,
        description="Full concatenated transcript of all segments in this chunk.",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Aggregate dense embedding vector for this audio chunk.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )
    source_path: Optional[str] = Field(
        default=None, description="Path to the originating audio source file."
    )


class MultimodalRetrievalResult(BaseModel):
    """Unified retrieval result that can carry any supported modality.

    A ``MultimodalRetrievalResult`` is returned by multimodal retrievers and
    stores the relevance score alongside whichever modality-specific payload
    matched the query.  Exactly one of ``image_chunk``, ``video_chunk``, or
    ``audio_chunk`` is expected to be populated for non-text modalities;
    for text results all three will be ``None``.

    Attributes:
        chunk_id: Identifier of the retrieved chunk.
        score: Relevance / similarity score (higher is more relevant).
        modality: The modality of the retrieved content.  One of:
            ``"text"``, ``"image"``, ``"video"``, ``"audio"``,
            ``"cross_modal"``.
        text: Textual representation of the result (transcript, caption, or
            the original passage for text modality).
        image_chunk: Populated when ``modality == "image"``.
        video_chunk: Populated when ``modality == "video"``.
        audio_chunk: Populated when ``modality == "audio"``.
        metadata: Arbitrary key-value metadata propagated from the source chunk.
        source: Human-readable label indicating the originating source
            (e.g. file path, URL, document title).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str = Field(..., description="Identifier of the retrieved chunk.")
    score: float = Field(..., description="Relevance/similarity score.")
    modality: Literal["text", "image", "video", "audio", "cross_modal"] = Field(
        ...,
        description=(
            "Modality of the retrieved content: 'text', 'image', 'video', "
            "'audio', or 'cross_modal'."
        ),
    )
    text: Optional[str] = Field(
        default=None,
        description=(
            "Textual representation of the result "
            "(transcript, caption, or original passage)."
        ),
    )
    image_chunk: Optional[ImageChunk] = Field(
        default=None, description="Image chunk payload (populated when modality='image')."
    )
    video_chunk: Optional[VideoChunk] = Field(
        default=None, description="Video chunk payload (populated when modality='video')."
    )
    audio_chunk: Optional[AudioChunk] = Field(
        default=None, description="Audio chunk payload (populated when modality='audio')."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )
    source: Optional[str] = Field(
        default=None,
        description="Human-readable label for the originating source.",
    )
