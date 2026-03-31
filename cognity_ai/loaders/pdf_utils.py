"""PDF utility functions for table/image extraction, metadata, slicing, merging, and rendering."""
from __future__ import annotations
import io
from pathlib import Path


def extract_tables(path: str) -> list[dict]:
    """Extract tables from all pages. Returns list of {page_num, table_data}."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    results = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in (tables or []):
                results.append({
                    "page_num": page_num,
                    "table_data": table,
                })
    return results


def extract_images(path: str) -> list[dict]:
    """Extract embedded images from all pages. Returns list of {page_num, image_bytes, mime_type}."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    results = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for img_info in (page.images or []):
                try:
                    x0 = img_info.get("x0", 0)
                    y0 = img_info.get("y0", 0)
                    x1 = img_info.get("x1", 0)
                    y1 = img_info.get("y1", 0)
                    cropped = page.crop((x0, y0, x1, y1))
                    img_bytes = cropped.to_image(resolution=150).original.tobytes("png")
                    results.append({
                        "page_num": page_num,
                        "image_bytes": img_bytes,
                        "mime_type": "image/png",
                    })
                except Exception:
                    pass
    return results


def extract_metadata(path: str) -> dict:
    """Extract document metadata: title, author, creation_date, page_count."""
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path) as pdf:
            meta = pdf.metadata or {}
            return {
                "title": meta.get("Title", ""),
                "author": meta.get("Author", ""),
                "creation_date": meta.get("CreationDate", ""),
                "page_count": len(pdf.pages),
            }
    except ImportError:
        pass

    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(path)
        info = reader.metadata or {}
        return {
            "title": info.get("/Title", ""),
            "author": info.get("/Author", ""),
            "creation_date": info.get("/CreationDate", ""),
            "page_count": len(reader.pages),
        }
    except ImportError:
        pass

    raise ImportError("Install pdfplumber: pip install pdfplumber")


def slice_pages(path: str, start: int, end: int) -> bytes:
    """Return a new PDF as bytes containing pages from start to end (1-indexed, inclusive)."""
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

    reader = PdfReader(path)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages, start=1):
        if start <= i <= end:
            writer.add_page(page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def merge_pdfs(paths: list[str]) -> bytes:
    """Merge multiple PDF files and return the result as bytes."""
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def pdf_to_images(path: str, dpi: int = 150) -> list[bytes]:
    """Render each PDF page as a PNG image. Returns list of PNG bytes per page."""
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    images = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                img = page.to_image(resolution=dpi)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images.append(buf.getvalue())
            except Exception:
                images.append(b"")
    return images
