"""Abstract base class for page/section index backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from cognity_ai.models.retrieval import PageInfo
from cognity_ai.utils.trie import Trie


class BasePageIndex(ABC):
    # Heading tries are lazily initialised so subclasses don't need to call
    # super().__init__().  Access always goes through the properties below.
    _heading_tries: dict[str, Trie]
    _heading_map: dict[str, dict[str, PageInfo]]

    @property
    def _htries(self) -> dict[str, Trie]:
        try:
            return self._heading_tries
        except AttributeError:
            self._heading_tries = {}
            return self._heading_tries

    @property
    def _hmap(self) -> dict[str, dict[str, PageInfo]]:
        try:
            return self._heading_map
        except AttributeError:
            self._heading_map = {}
            return self._heading_map

    @abstractmethod
    def detect_pages(self, text: str, doc_id: str, loader_metadata: dict | None = None) -> list[PageInfo]: ...

    @abstractmethod
    def get_page_for_char(self, doc_id: str, char_offset: int) -> PageInfo | None: ...

    @abstractmethod
    def store(self, doc_id: str, pages: list[PageInfo]): ...

    @abstractmethod
    def get(self, doc_id: str) -> list[PageInfo]: ...

    @abstractmethod
    def remove(self, doc_id: str): ...

    @abstractmethod
    def persist(self): ...

    # ── Helper called by subclass store() implementations ─────────────────

    def _build_heading_index(self, doc_id: str, pages: list[PageInfo]) -> None:
        """Build/rebuild the heading trie and map for *doc_id*."""
        trie: Trie = Trie()
        hmap: dict[str, PageInfo] = {}
        for p in pages:
            heading = (p.heading or "").lower()
            if heading:
                trie.insert(heading)
                hmap[heading] = p
        self._htries[doc_id] = trie
        self._hmap[doc_id] = hmap

    def _drop_heading_index(self, doc_id: str) -> None:
        self._htries.pop(doc_id, None)
        self._hmap.pop(doc_id, None)

    # ── Public helpers ────────────────────────────────────────────────────

    def get_page_text(self, doc_id: str, page_num: int, full_text: str) -> str | None:
        for p in self.get(doc_id):
            if p.page_num == page_num:
                return full_text[p.start_char:p.end_char]
        return None

    def get_section(self, doc_id: str, heading: str) -> PageInfo | None:
        """Return the PageInfo whose heading contains *heading* (case-insensitive).

        Uses the heading trie for O(k) prefix lookup when available, falling
        back to a linear scan for documents whose index hasn't been built yet.
        """
        key = heading.lower()
        trie = self._htries.get(doc_id)
        hmap = self._hmap.get(doc_id)
        if trie is not None and hmap is not None:
            # Try exact match first
            if key in hmap:
                return hmap[key]
            # Try prefix completion: if 'key' is a prefix of any stored heading
            matches = trie.words_with_prefix(key, max_results=1)
            if matches:
                return hmap.get(matches[0])
            return None
        # Fallback: linear scan (no trie built yet)
        for p in self.get(doc_id):
            if key in (p.heading or "").lower():
                return p
        return None
