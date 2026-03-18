"""LoaderFactory: maps file extensions to loader classes with lazy imports."""
from __future__ import annotations
from pathlib import Path

from raglib.loaders.base import BaseLoader
from raglib.models.document import Document


class LoaderFactory:
    """
    Central factory for obtaining the correct loader for a given file path.

    Loaders are imported lazily so that missing optional dependencies only raise
    errors when a loader for that format is actually requested.
    """

    _ext_map: dict[str, str] = {
        # Plain text
        ".txt": "text",
        # Markdown
        ".md": "markdown",
        ".markdown": "markdown",
        # PDF
        ".pdf": "pdf",
        # Word
        ".docx": "docx",
        ".doc": "docx",
        # Excel
        ".xlsx": "excel",
        ".xls": "excel",
        # CSV / TSV
        ".csv": "csv",
        ".tsv": "csv",
        # PowerPoint
        ".pptx": "pptx",
        ".ppt": "pptx",
        # HTML
        ".html": "html",
        ".htm": "html",
        # Structured data
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        # Images
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".bmp": "image",
        ".tiff": "image",
        ".tif": "image",
        ".webp": "image",
        ".gif": "image",
    }

    @classmethod
    def get_loader(cls, path: str, ocr_provider=None) -> BaseLoader:
        """Return an instantiated loader appropriate for the file at *path*."""
        ext = Path(path).suffix.lower()
        loader_name = cls._ext_map.get(ext)
        if loader_name is None:
            raise ValueError(
                f"No loader registered for extension '{ext}'. "
                f"Supported extensions: {sorted(cls._ext_map.keys())}"
            )

        if loader_name == "text":
            from raglib.loaders.text import TxtLoader
            return TxtLoader()

        if loader_name == "markdown":
            from raglib.loaders.text import MdLoader
            return MdLoader()

        if loader_name == "pdf":
            from raglib.loaders.pdf import PdfLoader
            return PdfLoader()

        if loader_name == "docx":
            from raglib.loaders.docx import DocxLoader
            return DocxLoader()

        if loader_name == "excel":
            from raglib.loaders.excel import ExcelLoader
            return ExcelLoader()

        if loader_name == "csv":
            from raglib.loaders.csv import CsvLoader
            return CsvLoader()

        if loader_name == "pptx":
            from raglib.loaders.pptx import PptxLoader
            return PptxLoader()

        if loader_name == "html":
            from raglib.loaders.html import HtmlLoader
            return HtmlLoader()

        if loader_name == "json":
            from raglib.loaders.json_loader import JsonLoader
            return JsonLoader()

        if loader_name == "yaml":
            from raglib.loaders.json_loader import YamlLoader
            return YamlLoader()

        if loader_name == "image":
            from raglib.loaders.image import ImageLoader
            return ImageLoader(ocr_provider=ocr_provider)

        raise ValueError(f"Internal error: unknown loader name '{loader_name}'")

    @classmethod
    def load(
        cls,
        path: str,
        doc_id: str | None = None,
        ocr_provider=None,
        **meta,
    ) -> list[Document]:
        """
        Convenience method: get the right loader, call load(), optionally
        override doc_id and attach extra metadata key-value pairs.
        """
        loader = cls.get_loader(path, ocr_provider=ocr_provider)
        docs = loader.load(path)

        if doc_id and docs:
            docs[0].doc_id = doc_id

        for doc in docs:
            if meta:
                doc.metadata.update(meta)

        return docs

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return all supported file extensions."""
        return sorted(cls._ext_map.keys())
