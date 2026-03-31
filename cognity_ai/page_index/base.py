"""Abstract base class for page/section index backends."""
from abc import ABC, abstractmethod
from cognity_ai.models.retrieval import PageInfo


class BasePageIndex(ABC):
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
