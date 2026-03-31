"""
OpenAI Whisper API (cloud) transcriber.

Install with: pip install openai

Requires a valid ``OPENAI_API_KEY`` environment variable or an explicit
*api_key* argument passed to the constructor.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Union

from raglib.multimodal.transcribers.base import (
    BaseTranscriber,
    TimestampedSegment,
    TranscriptionResult,
)


class WhisperAPITranscriber(BaseTranscriber):
    """Speech-to-text transcriber backed by the OpenAI Whisper cloud API.

    Sends audio to ``openai.audio.transcriptions.create`` and parses the
    ``verbose_json`` response (which includes segment-level timestamps).

    Parameters
    ----------
    api_key:
        OpenAI API key.  If omitted, the ``OPENAI_API_KEY`` environment
        variable is used automatically by the ``openai`` client.
    model:
        Whisper model name available through the OpenAI API.  Currently only
        ``"whisper-1"`` is available.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
    ) -> None:
        self.api_key = api_key
        self.model = model

    # ------------------------------------------------------------------
    # BaseTranscriber interface
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe *audio* via the OpenAI Whisper API.

        Parameters
        ----------
        audio:
            Filesystem path (``str`` / :class:`pathlib.Path`) to an audio file,
            or raw audio bytes (written to a temporary WAV file before upload).
        language:
            BCP-47 language code (e.g. ``"en"``) or ``"auto"`` for automatic
            language detection.

        Returns
        -------
        TranscriptionResult

        Raises
        ------
        ImportError
            If the ``openai`` package is not installed.
        openai.OpenAIError
            For API-level errors (invalid key, quota exceeded, etc.).
        """
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai is required for WhisperAPITranscriber. "
                "Install with: pip install openai"
            ) from exc

        client = openai.OpenAI(api_key=self.api_key)

        # Resolve bytes → temp file
        _tmp_path: Optional[str] = None
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                _tmp_path = tmp.name
            audio_path = _tmp_path
        else:
            audio_path = str(audio)

        try:
            with open(audio_path, "rb") as audio_file:
                # Pass language only when explicitly specified (not "auto")
                extra_kwargs: dict = {}
                if language not in ("auto", None):
                    extra_kwargs["language"] = language

                response = client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    **extra_kwargs,
                )
        finally:
            if _tmp_path is not None:
                Path(_tmp_path).unlink(missing_ok=True)

        # Parse response
        segments: list[TimestampedSegment] = []
        for seg in getattr(response, "segments", []) or []:
            # The verbose_json segment dict has start/end in seconds
            start_s = getattr(seg, "start", 0.0)
            end_s = getattr(seg, "end", 0.0)
            text = getattr(seg, "text", "").strip()
            segments.append(
                TimestampedSegment(
                    start_ms=start_s * 1000.0,
                    end_ms=end_s * 1000.0,
                    text=text,
                    confidence=1.0,
                )
            )

        full_text = getattr(response, "text", "").strip()
        detected_language = getattr(response, "language", language)
        duration_ms = getattr(response, "duration", 0.0) * 1000.0

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            language=detected_language,
            duration_ms=duration_ms,
            metadata={
                "backend": "whisper_api",
                "model": self.model,
            },
        )

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """Whisper API supports the same ~99 languages as the local model."""
        # Return "auto" and common language codes; the full list matches the
        # local Whisper model but is not independently enumerated by the API.
        return ["auto"]

    @property
    def supports_timestamps(self) -> bool:
        return True

    @property
    def supports_speaker_diarization(self) -> bool:
        return False
