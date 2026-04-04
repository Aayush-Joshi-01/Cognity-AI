"""StructuralPageIndex — builds PageInfo objects from loader-supplied page_map metadata."""
from cognity_ai.models.retrieval import PageInfo
from cognity_ai.page_index.base import BasePageIndex
from cognity_ai.page_index.json_store import JsonPageStore


class StructuralPageIndex(BasePageIndex):
    """Page index that reads pre-computed page boundaries from Document.page_map.

    When loader_metadata is provided (a list of dicts with keys
    page_num, start_char, end_char, and optionally heading/section),
    it converts them directly into PageInfo objects.  Falls back to a
    single-page result when no metadata is available.
    """

    def __init__(self, store_path: str = "./page_index.json"):
        self._store = JsonPageStore(store_path)

    # ── BasePageIndex interface ──────────────────────────────────────────

    def detect_pages(
        self,
        text: str,
        doc_id: str,
        loader_metadata=None,
    ) -> list[PageInfo]:
        """Build pages from loader_metadata and persist results.

        Args:
            text: Full document text (used for fallback end_char).
            doc_id: Document identifier used for storage.
            loader_metadata: List of dicts matching Document.page_map format
                [{page_num, start_char, end_char, heading?, section?}].
                When None or empty, falls back to single-page.

        Returns:
            List of PageInfo objects.
        """
        if loader_metadata:
            pages = [PageInfo(**p) for p in loader_metadata]
        else:
            pages = [PageInfo(page_num=1, start_char=0, end_char=len(text))]

        self._store.store(doc_id, pages)
        return pages

    def get_page_for_char(self, doc_id: str, char_offset: int) -> PageInfo | None:
        for p in self._store.get(doc_id):
            if p.start_char <= char_offset < p.end_char:
                return p
        return None

    def store(self, doc_id: str, pages: list[PageInfo]):
        self._store.store(doc_id, pages)
        self._build_heading_index(doc_id, pages)

    def get(self, doc_id: str) -> list[PageInfo]:
        return self._store.get(doc_id)

    def remove(self, doc_id: str):
        self._store.remove(doc_id)
        self._drop_heading_index(doc_id)

    def persist(self):
        self._store.save()
