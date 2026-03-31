"""
raglib.multimodal.pipeline.image_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image ingestion pipeline for multimodal RAG.

Flow::

  Image file → Pillow load → [Caption via BLIP-2 (optional)]
             → [OCR via raglib OCR subsystem (optional)]
             → Embed via multimodal embedder (CLIP/SigLIP)
             → Store in multimodal vector store

The pipeline is intentionally thin: it delegates captioning, OCR, embedding,
and storage to their respective subsystems, acting purely as an orchestrator.

.. warning::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.

Example::

    from cognity_ai.multimodal.embedders.clip import CLIPEmbedder
    from cognity_ai.multimodal.stores.chroma_mm import ChromaMultimodalStore
    from cognity_ai.multimodal.pipeline.image_pipeline import ImageIngestionPipeline

    embedder = CLIPEmbedder()
    store = ChromaMultimodalStore()
    pipeline = ImageIngestionPipeline(embedder=embedder, store=store)

    chunk = pipeline.ingest("photo.jpg", doc_id="doc-001")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

from cognity_ai.multimodal.models.media import ImageChunk
from cognity_ai.multimodal.stores.base import BaseMultimodalStore

logger = logging.getLogger(__name__)


class ImageIngestionPipeline:
    """End-to-end pipeline for ingesting image files into a multimodal store.

    The pipeline performs the following steps for each image:

    1. Load the image from disk using Pillow.
    2. Optionally generate a caption via a ``captioner`` (e.g. BLIP-2).
    3. Optionally extract text via an OCR provider from ``raglib.ocr``.
    4. Embed the image using a
       :class:`~raglib.multimodal.embedders.base.BaseMultimodalEmbedder`.
    5. Construct an :class:`~raglib.multimodal.models.media.ImageChunk` whose
       ``id`` is the SHA-256 hex digest of the canonical file path.
    6. Upsert the chunk into the provided
       :class:`~raglib.multimodal.stores.base.BaseMultimodalStore`.
    7. Return the chunk.

    Args:
        embedder: A
            :class:`~raglib.multimodal.embedders.base.BaseMultimodalEmbedder`
            instance (e.g. :class:`~raglib.multimodal.embedders.clip.CLIPEmbedder`
            or :class:`~raglib.multimodal.embedders.siglip.SigLIPEmbedder`).
        store: A
            :class:`~raglib.multimodal.stores.base.BaseMultimodalStore`
            instance into which ingested chunks are upserted.
        ocr: Optional
            :class:`~raglib.ocr.base.BaseOCR` instance.  When supplied, OCR is
            run on every image and the resulting text is stored in
            :attr:`~raglib.multimodal.models.media.ImageChunk.ocr_text`.
        captioner: Optional captioner that implements a
            ``caption_image(image_bytes_b64: str) -> str`` method (e.g.
            :class:`~raglib.multimodal.embedders.blip2.BLIP2Embedder`).  When
            supplied, a natural-language caption is generated and stored in
            :attr:`~raglib.multimodal.models.media.ImageChunk.caption`.
    """

    def __init__(
        self,
        embedder,
        store: BaseMultimodalStore,
        ocr=None,
        captioner=None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._ocr = ocr
        self._captioner = captioner

    # ── Public API ────────────────────────────────────────────────────────

    def ingest(
        self,
        image_path: str,
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ImageChunk:
        """Ingest a single image file.

        Args:
            image_path: Absolute or relative path to the image file.
                Supported formats: any format readable by Pillow (JPEG, PNG,
                WEBP, TIFF, BMP, …).
            doc_id: Optional parent document identifier.  Useful when the
                image was extracted from a larger document (PDF page, DOCX,
                …).
            metadata: Arbitrary key-value pairs that are merged into the
                :attr:`~raglib.multimodal.models.media.ImageChunk.metadata`
                field before the chunk is stored.

        Returns:
            The ingested :class:`~raglib.multimodal.models.media.ImageChunk`
            with ``embedding`` populated.

        Raises:
            FileNotFoundError: If *image_path* does not exist on disk.
            ValueError: If the embedder returns no embedding for the image.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path!r}")

        logger.debug("Ingesting image: %s", path)

        # Step 1 — load image bytes via Pillow and convert to base-64 JPEG
        image_bytes_b64 = _load_image_as_b64(path)

        # Step 2 — optional caption generation
        caption: Optional[str] = None
        if self._captioner is not None:
            try:
                caption = self._captioner.caption_image(image_bytes_b64)
                logger.debug("Generated caption for %s: %r", path.name, caption[:80])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Captioner failed for %s: %s", path.name, exc)

        # Step 3 — optional OCR
        ocr_text: Optional[str] = None
        if self._ocr is not None:
            try:
                ocr_text = self._ocr.ocr(path)
                logger.debug(
                    "OCR produced %d chars for %s", len(ocr_text or ""), path.name
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for %s: %s", path.name, exc)

        # Step 4 — embed
        embedding = self._embedder.embed_image(image_bytes_b64)
        if not embedding:
            raise ValueError(
                f"Embedder returned an empty embedding for image: {image_path!r}"
            )

        # Step 5 — build chunk
        chunk_id = _sha256_id(str(path))
        chunk = ImageChunk(
            id=chunk_id,
            doc_id=doc_id,
            image_path=str(path),
            image_bytes_b64=image_bytes_b64,
            caption=caption,
            ocr_text=ocr_text,
            embedding=embedding,
            page_num=(metadata or {}).get("page_num"),
            source_path=str(path),
            metadata=metadata or {},
        )

        # Step 6 — store
        self._store.upsert_image(chunk)
        logger.info("Ingested image chunk %s from %s", chunk_id, path.name)

        return chunk

    def ingest_batch(
        self,
        image_paths: list[str],
        doc_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[ImageChunk]:
        """Ingest multiple image files sequentially.

        Args:
            image_paths: List of paths to image files.
            doc_id: Optional shared parent document identifier applied to all
                images in the batch.
            metadata: Optional shared metadata applied to all images (merged
                per image; individual ``page_num`` values should be passed via
                per-image metadata by calling :meth:`ingest` directly).

        Returns:
            List of ingested
            :class:`~raglib.multimodal.models.media.ImageChunk` objects, in
            the same order as *image_paths*.  Failures are logged and skipped
            — the returned list may be shorter than *image_paths*.
        """
        chunks: list[ImageChunk] = []
        for path in image_paths:
            try:
                chunk = self.ingest(path, doc_id=doc_id, metadata=metadata)
                chunks.append(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest image %r: %s", path, exc)
        return chunks


# ── Module-level helpers ──────────────────────────────────────────────────────


def _load_image_as_b64(path: Path) -> str:
    """Load an image from disk, normalise to RGB JPEG, and return as base-64.

    Uses Pillow for decoding; the output is always a JPEG-encoded base-64
    string so that downstream embedders receive a consistent format regardless
    of the original file type.

    Args:
        path: Resolved path to the image file.

    Returns:
        Base-64-encoded JPEG bytes as a UTF-8 string.

    Raises:
        ImportError: If Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for ImageIngestionPipeline. "
            "Install with: pip install Pillow"
        ) from exc

    import base64
    import io

    with Image.open(path) as img:
        # Normalise palette / transparency modes before encoding
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        raw_bytes = buf.getvalue()

    return base64.b64encode(raw_bytes).decode("utf-8")


def _sha256_id(value: str) -> str:
    """Return the first 16 hex chars of the SHA-256 digest of *value*.

    Args:
        value: Input string (typically a resolved file path).

    Returns:
        16-character hex string suitable for use as a chunk ID.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
