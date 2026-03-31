"""Image file loader. Routes images through an optional OCR provider."""
from __future__ import annotations
import uuid
from pathlib import Path

from cognity_ai.loaders.base import BaseLoader
from cognity_ai.models.document import Document, ImageRef

_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"]


class ImageLoader(BaseLoader):
    """
    Routing loader for image files (.jpg, .png, etc.).

    If an ocr_provider (BaseOCR instance) is supplied, OCR is performed immediately
    and the result is stored in Document.text. Otherwise, text is left empty and
    the raw image bytes are stored in image_refs so that a downstream OCR step
    can populate the text later.
    """

    def __init__(self, ocr_provider=None):
        """
        Parameters
        ----------
        ocr_provider : BaseOCR | None
            An OCR provider instance. If None, no OCR is performed at load time.
        """
        self.ocr = ocr_provider

    @property
    def supported_extensions(self) -> list[str]:
        return _IMAGE_EXTENSIONS

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        img_bytes = p.read_bytes()
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

        # Determine MIME type from extension
        ext_to_mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = ext_to_mime.get(p.suffix.lower(), "image/png")

        # Run OCR immediately if a provider was supplied
        if self.ocr is not None:
            try:
                text = self.ocr.ocr(img_bytes)
            except Exception:
                text = ""
        else:
            text = ""

        image_ref = ImageRef(
            image_id=p.stem,
            char_offset=0,
            image_bytes=img_bytes,
            mime_type=mime_type,
            page_num=1,
            ocr_text=text,
        )

        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="ImageLoader",
                file_extension=p.suffix.lower(),
                file_size_bytes=p.stat().st_size,
                page_count=1,
                page_map=[{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}],
                image_refs=[image_ref],
            )
        ]
