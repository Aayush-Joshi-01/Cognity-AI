"""
Google Cloud Speech-to-Text v1 transcriber.

Install with: pip install google-cloud-speech

Authentication:
  - Provide a service-account JSON via *credentials_path*, or
  - Set the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable, or
  - Use Application Default Credentials (ADC) when running on GCP.

.. note::
    Files longer than 60 seconds must be uploaded to Google Cloud Storage
    (GCS) first.  This implementation sends short files inline and raises
    :class:`NotImplementedError` for long files unless a GCS URI is provided.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional, Union

from raglib.multimodal.transcribers.base import (
    BaseTranscriber,
    TimestampedSegment,
    TranscriptionResult,
)

# Maximum audio duration for synchronous (inline) requests
_SYNC_MAX_SECONDS = 60


class GoogleSTTTranscriber(BaseTranscriber):
    """Speech-to-text transcriber backed by Google Cloud Speech-to-Text v1.

    Short audio (≤ 60 s) is sent inline via a synchronous recognition request.
    Longer audio requires uploading to Google Cloud Storage and providing a
    ``gs://`` URI; currently a :class:`NotImplementedError` is raised in that
    case to prompt the caller to handle GCS upload externally.

    Parameters
    ----------
    credentials_path:
        Path to a service-account JSON credentials file.  When ``None``,
        Application Default Credentials (ADC) are used.
    project_id:
        Optional GCP project ID; used when initialising the client.
    model:
        Recognition model name.  Defaults to ``"latest_long"`` which is
        optimised for long-form audio.  Other options include
        ``"latest_short"``, ``"phone_call"``, ``"video"``, and
        ``"command_and_search"``.

    Raises
    ------
    ImportError
        If the ``google-cloud-speech`` package is not installed.
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "latest_long",
    ) -> None:
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.model = model

    # ------------------------------------------------------------------
    # BaseTranscriber interface
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language: str = "en-US",
    ) -> TranscriptionResult:
        """Transcribe *audio* using Google Cloud Speech-to-Text.

        Parameters
        ----------
        audio:
            Filesystem path or raw audio bytes.  For paths, the file is read
            into memory and sent inline (synchronous API) when ≤ 60 s.
        language:
            BCP-47 language code (e.g. ``"en-US"``, ``"fr-FR"``).  ``"auto"``
            falls back to ``"en-US"`` since Google STT requires an explicit
            language code.

        Returns
        -------
        TranscriptionResult

        Raises
        ------
        ImportError
            If ``google-cloud-speech`` is not installed.
        NotImplementedError
            If the audio exceeds 60 seconds (GCS upload required).
        """
        try:
            from google.cloud import speech  # type: ignore
            from google.oauth2 import service_account  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-cloud-speech is required for GoogleSTTTranscriber. "
                "Install with: pip install google-cloud-speech"
            ) from exc

        # Resolve language: "auto" is not supported, default to "en-US"
        if language in ("auto", None):
            language = "en-US"

        # Build credentials
        credentials = None
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        client_kwargs: dict = {}
        if credentials:
            client_kwargs["credentials"] = credentials
        if self.project_id:
            client_kwargs["client_options"] = {
                "quota_project_id": self.project_id
            }

        client = speech.SpeechClient(**client_kwargs)

        # Load audio bytes
        if isinstance(audio, bytes):
            audio_bytes = audio
        else:
            audio_bytes = Path(audio).read_bytes()

        # Estimate duration from byte size (rough heuristic for WAV 16kHz mono)
        estimated_seconds = len(audio_bytes) / (16000 * 2)  # 16-bit PCM

        if estimated_seconds > _SYNC_MAX_SECONDS:
            raise NotImplementedError(
                f"Audio appears to be longer than {_SYNC_MAX_SECONDS} seconds "
                f"(estimated {estimated_seconds:.0f} s). "
                "Please upload the file to GCS and use a GCS URI for long-form "
                "transcription via longrunningrecognize."
            )

        audio_content = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language,
            enable_word_time_offsets=True,
            model=self.model,
            enable_automatic_punctuation=True,
        )

        response = client.recognize(config=config, audio=audio_content)

        # Flatten word-level timestamps into sentence-level segments
        segments: list[TimestampedSegment] = []
        full_parts: list[str] = []

        for result in response.results:
            alt = result.alternatives[0]
            text = alt.transcript.strip()
            full_parts.append(text)

            if alt.words:
                start_ms = alt.words[0].start_time.total_seconds() * 1000.0
                end_ms = alt.words[-1].end_time.total_seconds() * 1000.0
            else:
                start_ms = 0.0
                end_ms = 0.0

            segments.append(
                TimestampedSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=float(alt.confidence),
                )
            )

        full_text = " ".join(full_parts)
        duration_ms = segments[-1].end_ms if segments else 0.0

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            language=language,
            duration_ms=duration_ms,
            metadata={
                "backend": "google_stt",
                "model": self.model,
            },
        )

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """Google STT supports 125+ languages; return common BCP-47 codes."""
        return [
            "af-ZA", "ar-EG", "bg-BG", "bn-BD", "ca-ES",
            "cs-CZ", "da-DK", "de-DE", "el-GR", "en-AU",
            "en-GB", "en-IN", "en-US", "es-ES", "es-MX",
            "et-EE", "fa-IR", "fi-FI", "fil-PH", "fr-CA",
            "fr-FR", "gu-IN", "hi-IN", "hr-HR", "hu-HU",
            "id-ID", "it-IT", "ja-JP", "kn-IN", "ko-KR",
            "lt-LT", "lv-LV", "ml-IN", "mr-IN", "ms-MY",
            "nl-NL", "no-NO", "pl-PL", "pt-BR", "pt-PT",
            "ro-RO", "ru-RU", "sk-SK", "sl-SI", "sr-RS",
            "sv-SE", "sw-KE", "ta-IN", "te-IN", "th-TH",
            "tr-TR", "uk-UA", "ur-PK", "vi-VN", "zh-CN",
            "zh-TW", "zu-ZA",
        ]

    @property
    def supports_timestamps(self) -> bool:
        return True

    @property
    def supports_speaker_diarization(self) -> bool:
        """Google STT supports speaker diarization via ``DiarizationConfig``."""
        return True
