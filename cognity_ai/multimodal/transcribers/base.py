"""
Abstract base class and shared data models for speech-to-text transcribers.

All concrete transcriber implementations inherit from :class:`BaseTranscriber`
and return :class:`TranscriptionResult` objects.
"""
from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field


class TimestampedSegment(BaseModel):
    """A single timestamped utterance within a transcription.

    Attributes:
        start_ms: Start time of this segment in milliseconds.
        end_ms: End time of this segment in milliseconds.
        text: Transcribed text for this segment.
        confidence: ASR confidence score in ``[0.0, 1.0]``.
        speaker_id: Optional diarization speaker label (e.g. ``"SPEAKER_00"``).
    """

    start_ms: float = Field(..., description="Segment start time in milliseconds.")
    end_ms: float = Field(..., description="Segment end time in milliseconds.")
    text: str = Field(..., description="Transcribed text for this segment.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="ASR confidence score in [0.0, 1.0].",
    )
    speaker_id: Optional[str] = Field(
        default=None, description="Diarization speaker label, if available."
    )


class TranscriptionResult(BaseModel):
    """The complete output of a transcription pass.

    Attributes:
        full_text: Concatenated transcript of all segments.
        segments: Ordered list of :class:`TimestampedSegment` objects.
        language: BCP-47 language code of the detected/specified language.
        duration_ms: Total audio duration in milliseconds.
        metadata: Arbitrary key-value metadata (model name, backend, etc.).
    """

    full_text: str = Field(..., description="Full concatenated transcript.")
    segments: list[TimestampedSegment] = Field(
        default_factory=list, description="Ordered timestamped segments."
    )
    language: str = Field(default="en", description="BCP-47 language code.")
    duration_ms: float = Field(
        default=0.0, description="Total audio duration in milliseconds."
    )
    metadata: dict = Field(
        default_factory=dict, description="Arbitrary metadata key-value pairs."
    )


class BaseTranscriber(ABC):
    """Abstract base class for all speech-to-text transcribers.

    Subclasses must implement :meth:`transcribe`.  Convenience wrappers
    :meth:`transcribe_file` and :meth:`transcribe_bytes` delegate to it.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio and return a :class:`TranscriptionResult`.

        Parameters
        ----------
        audio:
            Either a filesystem path (``str`` or :class:`pathlib.Path`) or raw
            audio bytes.
        language:
            BCP-47 language code (e.g. ``"en"``, ``"fr"``), or ``"auto"`` to
            let the backend detect the language automatically.

        Returns
        -------
        TranscriptionResult
        """
        ...

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def transcribe_file(self, path: str, language: str = "auto") -> TranscriptionResult:
        """Transcribe an audio file at the given filesystem *path*.

        This is a thin wrapper around :meth:`transcribe` that accepts only
        string paths for API clarity.

        Parameters
        ----------
        path:
            Absolute or relative path to the audio file.
        language:
            BCP-47 language code or ``"auto"``.

        Returns
        -------
        TranscriptionResult
        """
        return self.transcribe(audio=path, language=language)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        format: str = "wav",
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio provided as raw *bytes*.

        The default implementation writes the bytes to a temporary file and
        delegates to :meth:`transcribe`.  Subclasses may override this method
        for more efficient in-memory handling.

        Parameters
        ----------
        audio_bytes:
            Raw audio bytes.
        format:
            Audio container format (e.g. ``"wav"``, ``"mp3"``).
        language:
            BCP-47 language code or ``"auto"``.

        Returns
        -------
        TranscriptionResult
        """
        suffix = f".{format.lstrip('.')}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe(audio=tmp_path, language=language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """List of BCP-47 language codes supported by this transcriber.

        The default returns ``["auto"]``, meaning language detection is the
        only guaranteed mode.  Subclasses should override this to enumerate
        their supported languages.
        """
        return ["auto"]

    @property
    def supports_timestamps(self) -> bool:
        """Whether this transcriber can return segment-level timestamps.

        Defaults to ``True``; override to ``False`` in implementations that
        only return plain text without timing information.
        """
        return True

    @property
    def supports_speaker_diarization(self) -> bool:
        """Whether this transcriber supports speaker diarization.

        Defaults to ``False``; override to ``True`` in implementations that
        can assign speaker labels to segments.
        """
        return False
