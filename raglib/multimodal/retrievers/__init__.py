"""
raglib.multimodal.retrievers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multimodal retrieval strategies for the ``raglib.multimodal`` extension.

Exports the abstract base class and all concrete retriever implementations.
Heavy model dependencies are not imported at module level — only the class
objects are re-exported.

Available retrievers
--------------------
- :class:`BaseMultimodalRetriever` — Abstract base for all multimodal retrievers.
- :class:`ImageRetriever`          — Text-to-image and image-to-image retrieval (CLIP/SigLIP).
- :class:`VideoRetriever`          — Text-to-video-segment retrieval with timestamps.
- :class:`AudioRetriever`          — Text-to-audio-segment retrieval via transcripts or ImageBind.
- :class:`CrossModalRetriever`     — Any-to-any retrieval using a shared embedding space (ImageBind).

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from raglib.multimodal.retrievers.audio_retriever import AudioRetriever
from raglib.multimodal.retrievers.base import BaseMultimodalRetriever
from raglib.multimodal.retrievers.cross_modal import CrossModalRetriever
from raglib.multimodal.retrievers.image_retriever import ImageRetriever
from raglib.multimodal.retrievers.video_retriever import VideoRetriever

__all__ = [
    "BaseMultimodalRetriever",
    "ImageRetriever",
    "VideoRetriever",
    "AudioRetriever",
    "CrossModalRetriever",
]
