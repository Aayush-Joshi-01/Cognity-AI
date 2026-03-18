"""
AWS Transcribe transcriber (asynchronous job-based).

Install with: pip install boto3

AWS Transcribe is inherently asynchronous: audio must first be uploaded to
Amazon S3, then a transcription job is started, and results are polled until
the job completes.

Prerequisites:
  - An S3 bucket accessible to the caller (``s3_bucket`` parameter).
  - AWS credentials with ``transcribe:StartTranscriptionJob``,
    ``transcribe:GetTranscriptionJob``, ``s3:PutObject``, and
    ``s3:GetObject`` permissions.
"""
from __future__ import annotations

import json
import time
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union
from urllib import request as urllib_request

from raglib.multimodal.transcribers.base import (
    BaseTranscriber,
    TimestampedSegment,
    TranscriptionResult,
)

_DEFAULT_POLL_INTERVAL_S = 5
_DEFAULT_TIMEOUT_S = 300


class AWSTranscribeTranscriber(BaseTranscriber):
    """Speech-to-text transcriber backed by AWS Transcribe.

    Because AWS Transcribe processes jobs asynchronously, this implementation
    performs the following steps on every :meth:`transcribe` call:

    1. Upload the audio file to the configured S3 bucket.
    2. Start an ``AWS Transcribe`` job.
    3. Poll ``GetTranscriptionJob`` every 5 seconds until the job succeeds or
       fails (timeout: 300 s).
    4. Download the JSON transcript from S3 and parse word-level timestamps.
    5. Clean up the uploaded S3 object.

    Parameters
    ----------
    aws_region:
        AWS region where the Transcribe and S3 services will be called.
    s3_bucket:
        Name of the S3 bucket used for temporary audio and transcript storage.
        **Required** — AWS Transcribe cannot read local files directly.
    aws_access_key:
        AWS access key ID.  If ``None``, boto3 uses the standard credential
        chain (environment variables, ``~/.aws/credentials``, IAM role, etc.).
    aws_secret_key:
        AWS secret access key corresponding to *aws_access_key*.

    Raises
    ------
    ImportError
        If the ``boto3`` package is not installed.
    ValueError
        If *s3_bucket* is not provided.
    """

    def __init__(
        self,
        aws_region: str = "us-east-1",
        s3_bucket: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
    ) -> None:
        if not s3_bucket:
            raise ValueError(
                "s3_bucket is required for AWSTranscribeTranscriber. "
                "Provide the name of an S3 bucket the caller has read/write access to."
            )
        self.aws_region = aws_region
        self.s3_bucket = s3_bucket
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key

    # ------------------------------------------------------------------
    # BaseTranscriber interface
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, bytes, Path],
        language: str = "en-US",
    ) -> TranscriptionResult:
        """Transcribe *audio* using AWS Transcribe.

        Parameters
        ----------
        audio:
            Filesystem path or raw audio bytes.  Raw bytes are written to a
            temporary WAV file before being uploaded to S3.
        language:
            BCP-47 language code (e.g. ``"en-US"``).  ``"auto"`` falls back to
            ``"en-US"`` since AWS Transcribe requires an explicit language code
            (unless IdentifyLanguage is enabled, which is not used here for
            simplicity).

        Returns
        -------
        TranscriptionResult

        Raises
        ------
        ImportError
            If ``boto3`` is not installed.
        RuntimeError
            If the transcription job fails or times out.
        """
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for AWSTranscribeTranscriber. "
                "Install with: pip install boto3"
            ) from exc

        # Resolve language
        if language in ("auto", None):
            language = "en-US"

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
            return self._run_transcription_job(
                boto3=boto3,
                audio_path=audio_path,
                language=language,
            )
        finally:
            if _tmp_path is not None:
                Path(_tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supported_languages(self) -> list[str]:
        """AWS Transcribe supports 30+ language codes."""
        return [
            "af-ZA", "ar-AE", "ar-SA", "zh-CN", "zh-TW",
            "da-DK", "nl-NL", "en-AU", "en-GB", "en-IN",
            "en-IE", "en-NZ", "en-AB", "en-ZA", "en-US",
            "en-WL", "fr-FR", "fr-CA", "fa-IR", "de-DE",
            "de-CH", "he-IL", "hi-IN", "id-ID", "it-IT",
            "ja-JP", "ko-KR", "ms-MY", "pt-PT", "pt-BR",
            "ru-RU", "es-ES", "es-US", "sv-SE", "tl-PH",
            "ta-IN", "te-IN", "th-TH", "tr-TR", "uk-UA",
            "vi-VN",
        ]

    @property
    def supports_timestamps(self) -> bool:
        return True

    @property
    def supports_speaker_diarization(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _boto3_session(self, boto3):
        """Build a boto3 session with optional explicit credentials."""
        kwargs: dict = {"region_name": self.aws_region}
        if self.aws_access_key and self.aws_secret_key:
            kwargs["aws_access_key_id"] = self.aws_access_key
            kwargs["aws_secret_access_key"] = self.aws_secret_key
        return boto3.Session(**kwargs)

    def _run_transcription_job(
        self,
        boto3,
        audio_path: str,
        language: str,
    ) -> TranscriptionResult:
        """Upload audio, run the Transcribe job, and parse the result."""
        session = self._boto3_session(boto3)
        s3 = session.client("s3")
        transcribe = session.client("transcribe")

        job_id = "raglib-" + uuid.uuid4().hex
        s3_key = f"raglib-transcribe/{job_id}/{Path(audio_path).name}"
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"

        # --- 1. Upload to S3 ------------------------------------------
        s3.upload_file(audio_path, self.s3_bucket, s3_key)

        try:
            # --- 2. Start transcription job ---------------------------
            transcribe.start_transcription_job(
                TranscriptionJobName=job_id,
                Media={"MediaFileUri": s3_uri},
                MediaFormat=Path(audio_path).suffix.lstrip(".") or "wav",
                LanguageCode=language,
                Settings={
                    "ShowSpeakerLabels": False,
                    "ShowAlternatives": False,
                },
            )

            # --- 3. Poll until complete --------------------------------
            elapsed = 0
            while elapsed < _DEFAULT_TIMEOUT_S:
                status_resp = transcribe.get_transcription_job(
                    TranscriptionJobName=job_id
                )
                job = status_resp["TranscriptionJob"]
                job_status = job["TranscriptionJobStatus"]

                if job_status == "COMPLETED":
                    break
                if job_status == "FAILED":
                    reason = job.get("FailureReason", "unknown reason")
                    raise RuntimeError(
                        f"AWS Transcribe job {job_id!r} failed: {reason}"
                    )

                time.sleep(_DEFAULT_POLL_INTERVAL_S)
                elapsed += _DEFAULT_POLL_INTERVAL_S
            else:
                raise RuntimeError(
                    f"AWS Transcribe job {job_id!r} timed out after "
                    f"{_DEFAULT_TIMEOUT_S} seconds."
                )

            # --- 4. Download and parse transcript ---------------------
            transcript_uri = (
                job["Transcript"]["TranscriptFileUri"]
            )
            with urllib_request.urlopen(transcript_uri) as resp:
                transcript_data = json.loads(resp.read().decode("utf-8"))

        finally:
            # --- 5. Clean up S3 object --------------------------------
            try:
                s3.delete_object(Bucket=self.s3_bucket, Key=s3_key)
            except Exception:
                pass  # Best-effort cleanup

        return self._parse_transcript(transcript_data, language)

    def _parse_transcript(
        self,
        transcript_data: dict,
        language: str,
    ) -> TranscriptionResult:
        """Parse the AWS Transcribe JSON transcript into a :class:`TranscriptionResult`."""
        results = transcript_data.get("results", {})
        full_text = (
            results.get("transcripts", [{}])[0].get("transcript", "").strip()
        )

        items = results.get("items", [])

        # Group pronunciation items into sentence-like segments using punctuation
        # boundaries as delimiters.
        segments: list[TimestampedSegment] = []
        current_words: list[dict] = []

        def _flush_segment() -> None:
            if not current_words:
                return
            text = " ".join(
                w["alternatives"][0]["content"]
                for w in current_words
                if w.get("type") == "pronunciation"
            ).strip()
            if not text:
                return
            start_ms = float(current_words[0].get("start_time", 0.0)) * 1000.0
            end_ms = float(current_words[-1].get("end_time", 0.0)) * 1000.0
            confidence_values = [
                float(w["alternatives"][0].get("confidence", 1.0))
                for w in current_words
                if w.get("type") == "pronunciation" and w.get("alternatives")
            ]
            avg_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else 1.0
            )
            segments.append(
                TimestampedSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=avg_confidence,
                )
            )
            current_words.clear()

        for item in items:
            if item.get("type") == "punctuation":
                content = item["alternatives"][0].get("content", "")
                # Use sentence-ending punctuation as a segment boundary
                if content in (".", "!", "?") and current_words:
                    _flush_segment()
            else:
                current_words.append(item)

        # Flush any remaining words
        _flush_segment()

        duration_ms = segments[-1].end_ms if segments else 0.0

        return TranscriptionResult(
            full_text=full_text,
            segments=segments,
            language=language,
            duration_ms=duration_ms,
            metadata={"backend": "aws_transcribe"},
        )
