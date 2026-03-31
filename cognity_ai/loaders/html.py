"""HTML file loader using BeautifulSoup4."""
from __future__ import annotations
import uuid
from pathlib import Path

from cognity_ai.loaders.base import BaseLoader
from cognity_ai.models.document import Document


class HtmlLoader(BaseLoader):
    """
    Loads .html/.htm files. Strips script/style tags and extracts
    text content. Builds page_map from heading tags (h1-h3).
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def load(self, path: str) -> list[Document]:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required to load HTML files. "
                "Install it with: pip install beautifulsoup4"
            )

        p = Path(path)
        raw_html = p.read_text(encoding="utf-8", errors="replace")
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove script and style tags entirely
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract <title>
        title_tag = soup.find("title")
        doc_title = title_tag.get_text(strip=True) if title_tag else p.stem

        # Build text and page_map from heading structure
        text, page_map = self._extract_text_and_sections(soup)

        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="HtmlLoader",
                file_extension=p.suffix.lower(),
                file_size_bytes=p.stat().st_size,
                page_count=max(len(page_map), 1),
                page_map=page_map,
                metadata={"title": doc_title},
            )
        ]

    def _extract_text_and_sections(self, soup) -> tuple[str, list[dict]]:
        """Walk the document and split into sections at h1/h2/h3 boundaries."""
        from bs4 import NavigableString, Tag  # type: ignore

        HEADING_TAGS = {"h1", "h2", "h3"}

        body = soup.find("body") or soup
        lines: list[str] = []
        page_map: list[dict] = []
        section_num = 0
        section_start = 0
        current_heading = ""

        def flush_section(end: int) -> None:
            nonlocal section_num
            if end > section_start:
                page_map.append({
                    "page_num": section_num,
                    "start_char": section_start,
                    "end_char": end,
                    "heading": current_heading,
                })
                section_num += 1

        char_offset = 0

        for element in body.descendants:
            if not isinstance(element, Tag):
                continue
            tag_name = element.name
            if tag_name not in HEADING_TAGS:
                continue
            # We have a heading — flush previous section
            heading_text = element.get_text(separator=" ", strip=True)
            flush_section(char_offset)
            current_heading = heading_text
            section_start = char_offset

        # Fall back: just get all text
        full_text = body.get_text(separator="\n", strip=True)
        # Clean up excessive blank lines
        cleaned_lines = []
        blank_run = 0
        for line in full_text.splitlines():
            if line.strip() == "":
                blank_run += 1
                if blank_run <= 1:
                    cleaned_lines.append("")
            else:
                blank_run = 0
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Build page_map from headings found in the rendered text
        page_map = self._build_page_map_from_text(text, soup)

        return text, page_map

    def _build_page_map_from_text(self, text: str, soup) -> list[dict]:
        """Build page_map by finding heading text positions within the extracted text."""
        HEADING_TAGS = ["h1", "h2", "h3"]
        headings = []
        for tag in soup.find_all(HEADING_TAGS):
            heading_text = tag.get_text(separator=" ", strip=True)
            if heading_text:
                headings.append(heading_text)

        if not headings:
            return [{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}]

        page_map = []
        section_num = 1
        search_start = 0
        heading_positions = []

        for heading in headings:
            pos = text.find(heading, search_start)
            if pos != -1:
                heading_positions.append((pos, heading))
                search_start = pos + len(heading)

        if not heading_positions:
            return [{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}]

        # Include preamble before first heading
        if heading_positions[0][0] > 0:
            page_map.append({
                "page_num": 0,
                "start_char": 0,
                "end_char": heading_positions[0][0],
                "heading": "",
            })

        for i, (pos, heading) in enumerate(heading_positions):
            end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(text)
            page_map.append({
                "page_num": section_num,
                "start_char": pos,
                "end_char": end,
                "heading": heading,
            })
            section_num += 1

        return page_map
