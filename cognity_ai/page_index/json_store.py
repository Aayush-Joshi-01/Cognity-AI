"""JsonPageStore — persistent JSON-backed store mapping doc_id → list[PageInfo]."""
import json
from pathlib import Path

from cognity_ai.models.retrieval import PageInfo


class JsonPageStore:
    """Stores and retrieves page/section metadata as JSON on disk."""

    def __init__(self, path: str = "./page_index.json"):
        self.path = Path(path)
        self._index: dict[str, list[dict]] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )

    # ── CRUD ─────────────────────────────────────────────────────────────

    def store(self, doc_id: str, pages: list[PageInfo]):
        self._index[doc_id] = [p.model_dump() for p in pages]
        self.save()

    def get(self, doc_id: str) -> list[PageInfo]:
        raw = self._index.get(doc_id, [])
        return [PageInfo(**p) for p in raw]

    def remove(self, doc_id: str):
        self._index.pop(doc_id, None)
        self.save()

    def has(self, doc_id: str) -> bool:
        return doc_id in self._index

    # ── Convenience helpers ───────────────────────────────────────────────

    def get_page_text(self, doc_id: str, page_num: int, full_text: str) -> str | None:
        for p in self.get(doc_id):
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
            "sections": [
                {"page": p.page_num, "heading": p.heading}
                for p in pages
                if p.heading
            ],
        }
