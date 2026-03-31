"""Plain text and Markdown file loaders."""
from __future__ import annotations
import re
import uuid
from pathlib import Path

from cognity_ai.loaders.base import BaseLoader
from cognity_ai.models.document import Document


class TxtLoader(BaseLoader):
    """Loads plain .txt files as a single Document."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]
        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="TxtLoader",
                file_extension=p.suffix.lower(),
                file_size_bytes=p.stat().st_size,
                page_count=1,
                page_map=[{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}],
            )
        ]


class MdLoader(BaseLoader):
    """Loads Markdown files. Parses heading structure into page_map."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

        page_map = self._build_page_map(text)

        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="MdLoader",
                file_extension=p.suffix.lower(),
                file_size_bytes=p.stat().st_size,
                page_count=max(len(page_map), 1),
                page_map=page_map,
            )
        ]

    def _build_page_map(self, text: str) -> list[dict]:
        """Parse markdown headings (#, ##, ###) to build page_map sections."""
        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}]

        page_map = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = match.group(2).strip()
            page_map.append({
                "page_num": i + 1,
                "start_char": start,
                "end_char": end,
                "heading": heading,
            })

        # Include any text before the first heading as page 0
        if matches[0].start() > 0:
            preamble = {
                "page_num": 0,
                "start_char": 0,
                "end_char": matches[0].start(),
                "heading": "",
            }
            page_map.insert(0, preamble)

        return page_map
