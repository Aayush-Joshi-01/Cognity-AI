"""
Local Whisper transcriber using the ``openai-whisper`` library.

Install with: pip install raglib[whisper]  # openai-whisper

.. note::
    The model is loaded lazily on the first call to :meth:`transcribe` and then
    cached for subsequent calls.  Loading a large model (e.g. ``"large-v3"``)
    can take tens of seconds on the first invocation.
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

# Whisper supports 99 languages; list the BCP-47 codes.
_WHISPER_LANGUAGES: list[str] = [
    "auto",
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "eo",
    "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha",
    "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it",
    "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln",
    "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms",
    "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps",
    "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so",
    "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk",
    "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh",
    "zu",
]

_VALID_MODEL_SIZES = [
    "tiny", "base", "small", "medium",
    "large", "large-v2", "large-v3",
]


class WhisperLocalTranscriber(BaseTranscriber):
    """Speech-to-text transcriber using the local ``openai-whisper`` model.

    The underlying Whisper model is loaded lazily the first time
    :meth:`transcribe` is called so that import time stays low even when the
    class is instantiated in an environment where the model weights are not
    yet cached.

    Parameters
    ----------
    model_size:
        Whisper model variant to use.  Must be one of ``"tiny"``, ``"base"``,
        ``"small"``, ``"medium"``, ``"large"``, ``"large-v2"``, or
        ``"large-v3"``.  Larger models are slower but more accurate.
    device:
        Torch device to load the model on.  ``"auto"`` selects CUDA if
        available, otherwise falls back to CPU.
    language:
        Fix the transcription language.  ``None`` lets Whisper detect the
        language automatically from the first 30 s of audio.

    Raises
    ------
    ImportError
        If ``openai-whisper`` is not installed.
    ValueError
        If *model_size* is not a recognised Whisper model name.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: Optional[str] = None,
    ) -> None:
        if model_size not in _VALID_MODEL_SIZES:
            raise ValueError(
                f"Invalid model_size {model_size!r}. "
                f"Must be one of: {_VALID_MODEL_SIZES}"
            )
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None  # lazy-loaded on first transcribe call

    # ------------------------------------------------------------------
    # BaseTranscriber interface
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe *audio* with the local Whisper model.

        Parameters
        ----------
        audio:
            Filesystem path (``str`` / :class:`pathlib.Path`) to an audio file,
            or raw audio bytes.
        language:
            BCP-47 language code or ``"auto"`` to enable automatic language
            detection.  Overrides the instance-level ``language`` attribute if
            not ``"auto"``.

        Returns
        -------
        TranscriptionResult
        """
        model = self._get_model()

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
            # Determine language override
            lang_override = self.language
            if language != "auto":
                lang_override = language
            # Whisper expects None for auto-detect
            whisper_lang = None if lang_override in (None, "auto") else lang_override

            result = model.transcribe(
                audio_path,
                language=whisper_lang,
                task="transcribe",
                verbose=False,
            )
        finally:
            if _tmp_path is not None:
                Path(_tmp_path).unlink(missing_ok=True)

        # Build segments
        segments: list[TimestampedSegment] = []
        for seg in result.get("segments", []):
            segments.append(
                TimestampedSegment(
                    start_ms=seg["start"] * 1000.0,
                    end_ms=seg["end"] * 1000.0,
                    text=seg["text"].strip(),
                    confidence=float(seg.get("avg_logprob", 0.0)),
                )
            )

        full_text = result.get("text", "").strip()
        detected_language = result.get("language", language)
        duration_ms = 0.0
        if segments:
            duration_ms = segments[-1].end_ms

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            language=detected_language,
            duration_ms=duration_ms,
            metadata={
                "backend": "whisper_local",
                "model_size": self.model_size,
                "device": self._resolve_device(),
            },
        )

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """All 99 languages supported by Whisper, plus ``"auto"``."""
        return _WHISPER_LANGUAGES

    @property
    def supports_timestamps(self) -> bool:
        return True

    @property
    def supports_speaker_diarization(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"``."""
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_model(self):
        """Lazily load and cache the Whisper model."""
        if self._model is not None:
            return self._model

        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for WhisperLocalTranscriber. "
                "Install with: pip install raglib[whisper]  # openai-whisper"
            ) from exc

        device = self._resolve_device()
        self._model = whisper.load_model(self.model_size, device=device)
        return self._model
