"""
raglib.multimodal — Experimental Multimodal RAG Extension
==========================================================

.. warning::
    This subpackage is **experimental** (beta). APIs may change between versions.
    Feedback and contributions are welcome.

Supports three additional modalities beyond text:

* **Image RAG** — CLIP / SigLIP / ImageBind embedders, text-to-image retrieval
* **Video RAG** — Frame extraction, scene detection, transcript alignment, temporal retrieval
* **Audio RAG** — Whisper / Google STT / AWS Transcribe, segment-level retrieval

Quick start::

    from raglib.multimodal import ImageIngestionPipeline, ImageRetriever
    from raglib.multimodal.embedders import CLIPEmbedder
    from raglib.multimodal.stores import ChromaMultimodalStore

    embedder = CLIPEmbedder()
    store = ChromaMultimodalStore()

    pipeline = ImageIngestionPipeline(embedder=embedder, store=store)
    pipeline.ingest("photo.jpg")

    retriever = ImageRetriever(embedder=embedder, store=store)
    results = retriever.retrieve("a dog playing in the park")

Install optional dependencies::

    pip install raglib[clip]          # CLIP embedder
    pip install raglib[siglip]        # SigLIP embedder
    pip install raglib[video]         # Video loading
    pip install raglib[audio]         # Audio loading
    pip install raglib[whisper]       # Local Whisper transcription
    pip install raglib[multimodal]    # Everything above
"""

from raglib.multimodal.models.media import (
    AudioChunk,
    AudioSegment,
    ImageChunk,
    MultimodalRetrievalResult,
    VideoChunk,
    VideoFrame,
)
from raglib.multimodal.embedders.base import BaseMultimodalEmbedder
from raglib.multimodal.transcribers.base import (
    BaseTranscriber,
    TimestampedSegment,
    TranscriptionResult,
)
from raglib.multimodal.stores.base import BaseMultimodalStore
from raglib.multimodal.retrievers.base import BaseMultimodalRetriever
from raglib.multimodal.pipeline import (
    AudioIngestionPipeline,
    ImageIngestionPipeline,
    VideoIngestionPipeline,
)
from raglib.multimodal.retrievers import (
    AudioRetriever,
    CrossModalRetriever,
    ImageRetriever,
    VideoRetriever,
)

__all__ = [
    # Models
    "ImageChunk",
    "VideoFrame",
    "VideoChunk",
    "AudioSegment",
    "AudioChunk",
    "MultimodalRetrievalResult",
    # Base classes
    "BaseMultimodalEmbedder",
    "BaseTranscriber",
    "TranscriptionResult",
    "TimestampedSegment",
    "BaseMultimodalStore",
    "BaseMultimodalRetriever",
    # Pipelines
    "ImageIngestionPipeline",
    "VideoIngestionPipeline",
    "AudioIngestionPipeline",
    # Retrievers
    "ImageRetriever",
    "VideoRetriever",
    "AudioRetriever",
    "CrossModalRetriever",
]
