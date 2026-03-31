"""
raglib.multimodal.retrievers.image_retriever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image retriever for multimodal RAG.

Supports:
  - Text-to-image: text query → find visually relevant images (CLIP/SigLIP)
  - Image-to-image: image query → find similar images
  - Combined: text + image → cross-modal search

Typical usage::

    from cognity_ai.multimodal.retrievers import ImageRetriever
    from cognity_ai.multimodal.embedders import CLIPEmbedder
    from cognity_ai.multimodal.stores import ChromaMultimodalStore

    embedder = CLIPEmbedder()
    store = ChromaMultimodalStore()

    retriever = ImageRetriever(embedder=embedder, store=store)

    # Text-to-image
    results = retriever.retrieve("a dog playing in a park")

    # Image-to-image
    results = retriever.retrieve_by_image("query.jpg")

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from cognity_ai.multimodal.models.media import MultimodalRetrievalResult
from cognity_ai.multimodal.retrievers.base import BaseMultimodalRetriever

if TYPE_CHECKING:
    from cognity_ai.generators.base import BaseGenerator
    from cognity_ai.multimodal.embedders.base import BaseMultimodalEmbedder
    from cognity_ai.multimodal.stores.base import BaseMultimodalStore


class ImageRetriever(BaseMultimodalRetriever):
    """Retriever for image modality using CLIP, SigLIP, or compatible embedders.

    Converts text or image queries into embedding vectors and performs
    nearest-neighbour search against an indexed collection of
    :class:`~raglib.multimodal.models.media.ImageChunk` objects.

    Parameters
    ----------
    embedder:
        A multimodal embedder that implements at least ``embed_text`` and
        ``embed_image`` (e.g. :class:`~raglib.multimodal.embedders.clip.CLIPEmbedder`).
    store:
        A multimodal vector store that implements ``query_images``.
    generator:
        Optional text generator used to synthesise a fluent answer from
        image captions and OCR text.  When ``None``, :meth:`query` returns
        a formatted list of captions instead.
    """

    def __init__(
        self,
        embedder: "BaseMultimodalEmbedder",
        store: "BaseMultimodalStore",
        generator: Optional["BaseGenerator"] = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    @property
    def modality(self) -> str:
        """Returns ``"image"``."""
        return "image"

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most visually relevant images for a text query.

        Embeds *query* using the configured text encoder (shared CLIP/SigLIP
        embedding space) and performs an approximate nearest-neighbour search
        over the image index.

        Parameters
        ----------
        query:
            Natural-language description of the desired image content.
        top_k:
            Maximum number of image results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of results with ``modality="image"``, most relevant
            first.  Each result's ``image_chunk`` field is populated.
        """
        embedding: list[float] = self._embedder.embed_text(query)
        raw_results: list[MultimodalRetrievalResult] = self._store.query_images(
            embedding, top_k
        )
        # Ensure modality tag is set correctly on every result.
        return [
            result.model_copy(update={"modality": "image"})
            for result in raw_results
        ]

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def retrieve_by_image(
        self,
        image: Union[str, bytes, Path],
        top_k: int = 5,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve images similar to a query image (image-to-image search).

        Embeds *image* using the configured image encoder and performs an
        approximate nearest-neighbour search over the image index.

        Parameters
        ----------
        image:
            Query image as a filesystem path (``str`` or
            :class:`pathlib.Path`) or raw image bytes.
        top_k:
            Maximum number of image results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of results with ``modality="image"``, most relevant
            first.
        """
        embedding: list[float] = self._embedder.embed_image(image)
        raw_results: list[MultimodalRetrievalResult] = self._store.query_images(
            embedding, top_k
        )
        return [
            result.model_copy(update={"modality": "image"})
            for result in raw_results
        ]

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question about images using retrieved captions and OCR.

        Retrieves the most relevant images, then either generates a fluent
        answer (if a generator was provided) or returns a formatted list of
        captions and OCR excerpts.

        Parameters
        ----------
        question:
            Natural-language question about image content.
        top_k:
            Maximum number of images to retrieve.

        Returns
        -------
        str
            Generated or formatted answer grounded in the retrieved images.
        """
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return "No relevant images found."

        # Build context from captions and OCR text.
        context_parts: list[str] = []
        for idx, result in enumerate(results, start=1):
            chunk = result.image_chunk
            if chunk is None:
                caption = result.text or "(no description)"
                ocr = ""
            else:
                caption = chunk.caption or result.text or "(no caption)"
                ocr = chunk.ocr_text or ""

            entry = f"Image {idx}: {caption}"
            if ocr:
                entry += f"\n  OCR text: {ocr}"
            context_parts.append(entry)

        context = "\n\n".join(context_parts)

        if self._generator is not None:
            return self._generator.generate(question=question, context=context)

        return context
