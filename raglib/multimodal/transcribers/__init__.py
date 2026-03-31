from raglib.multimodal.transcribers.base import BaseTranscriber, TranscriptionResult, TimestampedSegment

__all__ = ["BaseTranscriber", "TranscriptionResult", "TimestampedSegment"]

_LAZY_MAP = {
    "WhisperLocalTranscriber": "raglib.multimodal.transcribers.whisper_local",
    "WhisperAPITranscriber": "raglib.multimodal.transcribers.whisper_api",
    "GoogleSTTTranscriber": "raglib.multimodal.transcribers.google_stt",
    "AWSTranscribeTranscriber": "raglib.multimodal.transcribers.aws_transcribe",
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        import importlib

        mod = importlib.import_module(_LAZY_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
