"""
Page Index — persistent mapping of doc_id → page/section structure.
Enables page-level retrieval, section-aware chunking, and
structural queries like "what's on page 5 of doc X".
"""

import json
from pathlib import Path
from models import PageInfo


class PageIndex:
    def __init__(self, path: str = "./page_index.json"):
        self.path = Path(path)
        self._index: dict[str, list[dict]] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def save(self):
        self.path.write_text(json.dumps(self._index, indent=2))

    def store(self, doc_id: str, pages: list[PageInfo]):
        self._index[doc_id] = [p.model_dump() for p in pages]
        self.save()

    def get(self, doc_id: str) -> list[PageInfo]:
        raw = self._index.get(doc_id, [])
        return [PageInfo(**p) for p in raw]

    def remove(self, doc_id: str):
        self._index.pop(doc_id, None)
        self.save()

    def get_page_text(self, doc_id: str, page_num: int, full_text: str) -> str | None:
        pages = self.get(doc_id)
        for p in pages:
            if p.page_num == page_num:
                return full_text[p.start_char:p.end_char]
        return None

    def get_section(self, doc_id: str, heading: str) -> PageInfo | None:
        for p in self.get(doc_id):
            if heading.lower() in (p.heading or "").lower():
                return p
        return None

    def summary(self, doc_id: str) -> dict:
        pages = self.get(doc_id)
        return {
            "doc_id": doc_id,
            "page_count": len(pages),
            "sections": [{"page": p.page_num, "heading": p.heading} for p in pages if p.heading],
        }
