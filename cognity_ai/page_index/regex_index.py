"""RegexPageIndex — page/section detection using regex patterns from NLPProcessor."""
import re

from cognity_ai.models.retrieval import PageInfo
from cognity_ai.page_index.base import BasePageIndex
from cognity_ai.page_index.json_store import JsonPageStore


class RegexPageIndex(BasePageIndex):
    """Detects pages and sections using the same regex logic as NLPProcessor.detect_pages().

    Handles:
    - Form feeds (\\f)
    - --- or === horizontal separators
    - "Page N" patterns
    - Markdown headings (# ## ###)
    - Numbered sections (1. Title...)

    Persists results via JsonPageStore.
    """

    # Same patterns as NLPProcessor.detect_pages()
    _PATTERNS = [
        r'\f',                          # form feed
        r'\n-{3,}\n',                   # --- separator
        r'\n={3,}\n',                   # === separator
        r'(?:^|\n)(?:Page\s+\d+)',      # "Page N"
        r'(?:^|\n)(#{1,3}\s+.+)',       # Markdown headings
        r'(?:^|\n)(\d+\.\s+[A-Z].+)',  # Numbered sections
    ]
    _COMBINED = "|".join(f"({p})" for p in _PATTERNS)

    def __init__(self, store_path: str = "./page_index.json"):
        self._store = JsonPageStore(store_path)

    # ── BasePageIndex interface ──────────────────────────────────────────

    def detect_pages(
        self,
        text: str,
        doc_id: str,
        loader_metadata: dict | None = None,
    ) -> list[PageInfo]:
        """Detect page/section boundaries using regex and persist results."""
        pages = self._regex_detect(text)
        self._store.store(doc_id, pages)
        return pages

    def get_page_for_char(self, doc_id: str, char_offset: int) -> PageInfo | None:
        for p in self._store.get(doc_id):
            if p.start_char <= char_offset < p.end_char:
                return p
        return None

    def store(self, doc_id: str, pages: list[PageInfo]):
        self._store.store(doc_id, pages)

    def get(self, doc_id: str) -> list[PageInfo]:
        return self._store.get(doc_id)

    def remove(self, doc_id: str):
        self._store.remove(doc_id)

    def persist(self):
        self._store.save()

    # ── Core regex logic (mirrors NLPProcessor.detect_pages) ────────────

    def _regex_detect(self, text: str) -> list[PageInfo]:
        splits = list(re.finditer(self._COMBINED, text, re.MULTILINE))

        if not splits:
            return [PageInfo(page_num=1, start_char=0, end_char=len(text))]

        pages = []
        prev_end = 0
        for i, match in enumerate(splits):
            heading = match.group(0).strip().lstrip("#").strip()
            pages.append(PageInfo(
                page_num=i + 1,
                section=heading[:100],
                heading=heading[:100],
                start_char=prev_end,
                end_char=match.start(),
            ))
            prev_end = match.start()

        # Last segment
        pages.append(PageInfo(
            page_num=len(splits) + 1,
            start_char=prev_end,
            end_char=len(text),
        ))
        return pages
