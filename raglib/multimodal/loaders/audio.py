"""
Audio loader for multimodal RAG.

Supports: .mp3, .wav, .flac, .ogg, .m4a, .aac, .wma

Dependencies (lazy imports):
  - librosa — audio loading and analysis
  - soundfile — audio I/O
  - pydub — audio format conversion (fallback)

Install: pip install raglib[audio]
"""
from __future__ import annotations

import base64
import io
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from raglib.multimodal.models.media import AudioChunk, AudioSegment

_SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]


# ---------------------------------------------------------------------------
# Internal helpers — one per backend
# ---------------------------------------------------------------------------


def _load_with_librosa(path: str) -> tuple:
    """Load audio via *librosa*.

    Returns ``(samples_float32, sample_rate, num_channels)``.
    ``samples_float32`` has shape ``(num_channels, num_samples)`` or
    ``(num_samples,)`` for mono.
    """
    import librosa  # type: ignore
    import numpy as np  # type: ignore

    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        channels = 1
        y = y[np.newaxis, :]  # (1, samples)
    else:
        channels = y.shape[0]
    return y, sr, channels


def _load_with_soundfile(path: str) -> tuple:
    """Load audio via *soundfile*.

    Returns ``(samples_float32, sample_rate, num_channels)``.
    """
    import soundfile as sf  # type: ignore
    import numpy as np  # type: ignore

    data, sr = sf.read(path, always_2d=True)  # (samples, channels)
    data = data.T  # (channels, samples)
    channels = data.shape[0]
    return data, sr, channels


def _load_with_pydub(path: str) -> tuple:
    """Load audio via *pydub* (fallback; requires ffmpeg on PATH).

    Returns ``(samples_float32, sample_rate, num_channels)``.
    """
    from pydub import AudioSegment as PydubSegment  # type: ignore
    import numpy as np  # type: ignore

    audio = PydubSegment.from_file(path)
    sr = audio.frame_rate
    channels = audio.channels
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # Normalise to [-1, 1]
    max_val = float(2 ** (8 * audio.sample_width - 1))
    samples = samples / max_val
    if channels > 1:
        samples = samples.reshape(-1, channels).T  # (channels, samples)
    else:
        samples = samples[np.newaxis, :]  # (1, samples)
    return samples, sr, channels


def _samples_to_wav_bytes(samples, sample_rate: int, channels: int) -> bytes:
    """Encode a float32 numpy array to WAV bytes (16-bit PCM)."""
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    # soundfile expects (samples, channels) for multi-channel
    if channels == 1:
        data = samples[0]
    else:
        data = samples.T  # (samples, channels)

    data = np.clip(data, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _samples_to_wav_bytes_pydub(samples, sample_rate: int, channels: int) -> bytes:
    """Encode float32 numpy samples to WAV bytes using pydub (soundfile fallback)."""
    from pydub import AudioSegment as PydubSegment  # type: ignore
    import numpy as np  # type: ignore

    # Convert float32 -> int16
    data = np.clip(samples, -1.0, 1.0)
    if channels == 1:
        data = data[0]
    else:
        data = data.T  # (samples, channels)

    pcm = (data * 32767).astype(np.int16)
    if channels > 1:
        pcm_bytes = pcm.tobytes()
    else:
        pcm_bytes = pcm.tobytes()

    seg = PydubSegment(
        data=pcm_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def _encode_wav_b64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class AudioLoader:
    """Load audio files into :class:`~raglib.multimodal.models.media.AudioChunk` objects.

    Audio is loaded with *librosa* (preferred), then *soundfile*, then *pydub*.
    Each loaded file is split into overlapping fixed-duration segments.

    Parameters
    ----------
    chunk_duration_ms:
        Target duration of each audio segment in milliseconds.  Defaults to
        ``30_000`` (30 seconds).
    overlap_ms:
        Overlap between consecutive segments in milliseconds.  Defaults to
        ``5_000`` (5 seconds).
    normalize:
        Whether to normalise audio amplitude to the ``[-1, 1]`` range before
        splitting.
    """

    supported_extensions: list[str] = _SUPPORTED_EXTENSIONS

    def __init__(
        self,
        chunk_duration_ms: int = 30_000,
        overlap_ms: int = 5_000,
        normalize: bool = True,
    ) -> None:
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_load(self, path: str) -> bool:
        """Return ``True`` if *path* has a supported audio extension."""
        return Path(path).suffix.lower() in self.supported_extensions

    def load(self, path: str) -> AudioChunk:
        """Load an audio file and return an :class:`AudioChunk`.

        Parameters
        ----------
        path:
            Absolute or relative filesystem path to the audio file.

        Returns
        -------
        AudioChunk
            Contains time-segmented audio (each segment as base64-encoded
            WAV bytes) and file metadata.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not supported.
        ImportError
            If none of librosa, soundfile, or pydub is available.
        """
        p = Path(path).resolve()

        # --- 1. Validate -------------------------------------------------
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")
        ext = p.suffix.lower()
        if ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported audio extension {ext!r}. "
                f"Supported: {self.supported_extensions}"
            )

        # --- 2. Load audio -----------------------------------------------
        samples, sample_rate, channels = self._load_audio(str(p))

        # --- 3. Metadata -------------------------------------------------
        total_samples = samples.shape[1]
        duration_ms = (total_samples / sample_rate) * 1000.0

        # --- 4. Normalize ------------------------------------------------
        if self.normalize:
            import numpy as np  # type: ignore

            peak = float(np.abs(samples).max())
            if peak > 0.0:
                samples = samples / peak

        # --- 5. Split into segments --------------------------------------
        audio_segments = self._split_into_segments(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            audio_id=p.stem,
        )

        # --- 6. Build AudioChunk -----------------------------------------
        chunk_id = p.stem + "_" + uuid.uuid4().hex[:8]
        return AudioChunk(
            id=chunk_id,
            audio_id=chunk_id,
            segments=audio_segments,
            full_transcript=None,
            metadata={
                "source_path": str(p),
                "duration_ms": duration_ms,
                "sample_rate": sample_rate,
                "channels": channels,
                "format": ext.lstrip("."),
            },
            source_path=str(p),
        )

    def load_bytes(self, audio_bytes: bytes, format: str = "wav") -> AudioChunk:
        """Load audio from raw bytes (e.g. an audio track extracted from a video).

        The bytes are written to a temporary file, then loaded via the normal
        :meth:`load` pipeline.

        Parameters
        ----------
        audio_bytes:
            Raw audio bytes in the given *format*.
        format:
            Audio format / container identifier (e.g. ``"wav"``, ``"mp3"``).

        Returns
        -------
        AudioChunk
            Equivalent to calling :meth:`load` on a file containing the same audio.
        """
        suffix = f".{format.lstrip('.')}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            chunk = self.load(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Overwrite source_path to indicate the bytes origin
        chunk.source_path = "<bytes>"
        chunk.metadata["source_path"] = "<bytes>"
        return chunk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> tuple:
        """Try librosa -> soundfile -> pydub in order.

        Returns ``(samples_float32, sample_rate, num_channels)``.
        """
        # Try librosa
        try:
            import librosa  # noqa: F401

            return _load_with_librosa(path)
        except ImportError:
            pass
        except Exception:
            pass

        # Try soundfile
        try:
            import soundfile  # noqa: F401

            return _load_with_soundfile(path)
        except ImportError:
            pass
        except Exception:
            pass

        # Try pydub
        try:
            from pydub import AudioSegment as _  # noqa: F401

            return _load_with_pydub(path)
        except ImportError:
            pass

        raise ImportError(
            "No audio backend found. Install one of: librosa, soundfile, or pydub.\n"
            "Recommended: pip install raglib[audio]"
        )

    def _encode_segment_wav(self, samples, sample_rate: int, channels: int) -> str:
        """Encode a samples slice to base64 WAV, using soundfile or pydub fallback."""
        try:
            import soundfile  # noqa: F401

            wav_bytes = _samples_to_wav_bytes(samples, sample_rate, channels)
        except ImportError:
            wav_bytes = _samples_to_wav_bytes_pydub(samples, sample_rate, channels)
        return _encode_wav_b64(wav_bytes)

    def _split_into_segments(
        self,
        samples,
        sample_rate: int,
        channels: int,
        audio_id: str,
    ) -> list[AudioSegment]:
        """Split *samples* into overlapping fixed-duration segments.

        Parameters
        ----------
        samples:
            Float32 numpy array of shape ``(channels, total_samples)``.
        sample_rate:
            Sample rate in Hz.
        channels:
            Number of audio channels.
        audio_id:
            Base identifier used when constructing segment IDs.

        Returns
        -------
        list[AudioSegment]
            Ordered list of segments covering the full audio track.
        """
        total_samples = samples.shape[1]
        chunk_samples = int((self.chunk_duration_ms / 1000.0) * sample_rate)
        overlap_samples = int((self.overlap_ms / 1000.0) * sample_rate)
        step_samples = max(1, chunk_samples - overlap_samples)

        segments: list[AudioSegment] = []
        start = 0
        seg_idx = 0

        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            seg_samples = samples[:, start:end]

            start_ms = (start / sample_rate) * 1000.0
            end_ms = (end / sample_rate) * 1000.0

            b64 = self._encode_segment_wav(seg_samples, sample_rate, channels)
            segments.append(
                AudioSegment(
                    id=f"{audio_id}_seg{seg_idx}_{uuid.uuid4().hex[:6]}",
                    audio_id=audio_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    metadata={
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "audio_bytes_b64": b64,
                    },
                )
            )
            seg_idx += 1

            if end == total_samples:
                break
            start += step_samples

        return segments
