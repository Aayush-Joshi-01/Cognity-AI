"""Document model — the output of any loader before chunking."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ImageRef(BaseModel):
    """Reference to an embedded image within a document."""
    image_id: str
    char_offset: int          # Position in the parent Document.text where image was found
    image_bytes: Optional[bytes] = None  # Raw image bytes for OCR
    mime_type: str = "image/png"
    page_num: int = 0
    caption: str = ""
    ocr_text: str = ""        # Populated after OCR processing


class Document(BaseModel):
    """Unified document model output by all loaders."""
    doc_id: str
    text: str
    source_path: str = ""
    source_name: str = ""
    loader: str = ""          # Which loader produced this
    metadata: dict = Field(default_factory=dict)
    # Page/section metadata from loader (structural)
    page_map: list[dict] = Field(default_factory=list)  # [{page_num, start_char, end_char, heading}]
    image_refs: list[ImageRef] = Field(default_factory=list)
    # File info
    file_extension: str = ""
    file_size_bytes: int = 0
    # Content stats
    page_count: int = 0
    char_count: int = 0

    def model_post_init(self, __context):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.file_extension and self.source_path:
            from pathlib import Path
            self.file_extension = Path(self.source_path).suffix.lower()
