"""
raglib.multimodal.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end ingestion pipelines for multimodal RAG.

Each pipeline covers one modality, orchestrating loading, optional captioning
/ transcription, embedding, and storage into a
:class:`~raglib.multimodal.stores.base.BaseMultimodalStore`.

.. warning::
    This subpackage is **experimental**. APIs may change without notice.
"""

from cognity_ai.multimodal.pipeline.image_pipeline import ImageIngestionPipeline
from cognity_ai.multimodal.pipeline.video_pipeline import VideoIngestionPipeline
from cognity_ai.multimodal.pipeline.audio_pipeline import AudioIngestionPipeline

__all__ = [
    "ImageIngestionPipeline",
    "VideoIngestionPipeline",
    "AudioIngestionPipeline",
]
