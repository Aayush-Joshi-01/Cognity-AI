"""HybridPageIndex — combines StructuralPageIndex and RegexPageIndex (DEFAULT)."""
from cognity_ai.models.retrieval import PageInfo
from cognity_ai.page_index.base import BasePageIndex
from cognity_ai.page_index.regex_index import RegexPageIndex
from cognity_ai.page_index.structural_index import StructuralPageIndex


class HybridPageIndex(BasePageIndex):
    """Default page index: uses structural metadata when available, regex otherwise.

    Decision logic:
    - If loader_metadata is provided and non-empty → StructuralPageIndex
    - Otherwise → RegexPageIndex
    """

    def __init__(self, store_path: str = "./page_index.json"):
        self.structural = StructuralPageIndex(store_path=store_path)
        self.regex = RegexPageIndex(store_path=store_path)

    # ── BasePageIndex interface ──────────────────────────────────────────

    def detect_pages(
        self,
        text: str,
        doc_id: str,
        loader_metadata=None,
    ) -> list[PageInfo]:
        """Detect pages using the best available strategy.

        Args:
            text: Full document text.
            doc_id: Document identifier for storage.
            loader_metadata: Optional page_map from the loader.

        Returns:
            List of PageInfo objects, persisted to the shared JSON store.
        """
        if loader_metadata:
            return self.structural.detect_pages(text, doc_id, loader_metadata)
        return self.regex.detect_pages(text, doc_id, None)

    def get_page_for_char(self, doc_id: str, char_offset: int) -> PageInfo | None:
        # Delegate to structural first; its store is the same path as regex's store
        result = self.structural.get_page_for_char(doc_id, char_offset)
        if result is not None:
            return result
        return self.regex.get_page_for_char(doc_id, char_offset)

    def store(self, doc_id: str, pages: list[PageInfo]):
        # Write to both so that whichever delegate is queried next finds the data
        self.structural.store(doc_id, pages)

    def get(self, doc_id: str) -> list[PageInfo]:
        # Both delegates share the same JsonPageStore path, so either works
        return self.structural.get(doc_id)

    def remove(self, doc_id: str):
        self.structural.remove(doc_id)
        # regex store shares the same file path, so the entry is already gone

    def persist(self):
        self.structural.persist()
