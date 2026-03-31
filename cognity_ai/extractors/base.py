"""Abstract base class for knowledge extractors."""
from abc import ABC, abstractmethod
from cognity_ai.models.knowledge import ExtractionResult


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, source_id: str = "") -> ExtractionResult: ...
