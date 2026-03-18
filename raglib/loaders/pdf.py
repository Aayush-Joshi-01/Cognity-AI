"""PDF file loader with pdfplumber (preferred), pypdf, and pdfminer fallbacks."""
from __future__ import annotations
import uuid
from pathlib import Path

from raglib.loaders.base import BaseLoader
from raglib.models.document import Document, ImageRef


def _load_with_pdfplumber(path: str) -> list[Document]:
    import pdfplumber  # type: ignore

    p = Path(path)
    doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

    all_text_parts: list[str] = []
    page_map: list[dict] = []
    image_refs: list[ImageRef] = []
    char_offset = 0

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            # Append separator between pages
            if all_text_parts:
                separator = "\n"
            else:
                separator = ""

            start_char = char_offset + len(separator)
            all_text_parts.append(separator + page_text)
            end_char = char_offset + len(separator) + len(page_text)

            page_map.append({
                "page_num": page_num,
                "start_char": start_char,
                "end_char": end_char,
                "heading": f"Page {page_num}",
            })

            # Extract embedded images from the page
            for img_info in (page.images or []):
                try:
                    # pdfplumber stores image dict; attempt to get bytes via bbox crop
                    x0 = img_info.get("x0", 0)
                    y0 = img_info.get("y0", 0)
                    x1 = img_info.get("x1", 0)
                    y1 = img_info.get("y1", 0)
                    cropped = page.crop((x0, y0, x1, y1))
                    img_bytes = cropped.to_image(resolution=150).original.tobytes("png")
                    image_id = f"{p.stem}_p{page_num}_img{len(image_refs)}"
                    image_refs.append(ImageRef(
                        image_id=image_id,
                        char_offset=start_char,
                        image_bytes=img_bytes,
                        mime_type="image/png",
                        page_num=page_num,
                    ))
                except Exception:
                    # Skip images that cannot be extracted
                    pass

            char_offset = end_char

    full_text = "".join(all_text_parts)

    return [
        Document(
            doc_id=doc_id,
            text=full_text,
            source_path=str(p.resolve()),
            source_name=p.name,
            loader="PdfLoader[pdfplumber]",
            file_extension=".pdf",
            file_size_bytes=p.stat().st_size,
            page_count=len(page_map),
            page_map=page_map,
            image_refs=image_refs,
        )
    ]


def _load_with_pypdf(path: str) -> list[Document]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore

    p = Path(path)
    doc_id = p.stem + "_" + uuid.uuid4().hex[:8]
    reader = PdfReader(path)

    all_text_parts: list[str] = []
    page_map: list[dict] = []
    char_offset = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        separator = "\n" if all_text_parts else ""
        start_char = char_offset + len(separator)
        all_text_parts.append(separator + page_text)
        end_char = char_offset + len(separator) + len(page_text)

        page_map.append({
            "page_num": page_num,
            "start_char": start_char,
            "end_char": end_char,
            "heading": f"Page {page_num}",
        })
        char_offset = end_char

    full_text = "".join(all_text_parts)

    return [
        Document(
            doc_id=doc_id,
            text=full_text,
            source_path=str(p.resolve()),
            source_name=p.name,
            loader="PdfLoader[pypdf]",
            file_extension=".pdf",
            file_size_bytes=p.stat().st_size,
            page_count=len(page_map),
            page_map=page_map,
        )
    ]


def _load_with_pdfminer(path: str) -> list[Document]:
    from pdfminer.high_level import extract_text  # type: ignore

    p = Path(path)
    doc_id = p.stem + "_" + uuid.uuid4().hex[:8]
    full_text = extract_text(path) or ""

    return [
        Document(
            doc_id=doc_id,
            text=full_text,
            source_path=str(p.resolve()),
            source_name=p.name,
            loader="PdfLoader[pdfminer]",
            file_extension=".pdf",
            file_size_bytes=p.stat().st_size,
            page_count=1,
            page_map=[{"page_num": 1, "start_char": 0, "end_char": len(full_text), "heading": ""}],
        )
    ]


class PdfLoader(BaseLoader):
    """
    PDF loader that tries pdfplumber first (preferred, gives page-level text and
    embedded images), then falls back to pypdf, then pdfminer.six.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, path: str) -> list[Document]:
        # Try pdfplumber
        try:
            import pdfplumber  # noqa: F401
            return _load_with_pdfplumber(path)
        except ImportError:
            pass
        except Exception as exc:
            # pdfplumber failed on this file; try next backend
            pass

        # Try pypdf
        try:
            try:
                import pypdf  # noqa: F401
            except ImportError:
                import PyPDF2  # noqa: F401
            return _load_with_pypdf(path)
        except ImportError:
            pass
        except Exception:
            pass

        # Try pdfminer.six
        try:
            import pdfminer  # noqa: F401
            return _load_with_pdfminer(path)
        except ImportError:
            pass

        raise ImportError(
            "No PDF backend found. Install pdfplumber: pip install pdfplumber"
        )
