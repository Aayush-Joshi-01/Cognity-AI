"""Abstract base class for all document loaders."""
from abc import ABC, abstractmethod
from cognity_ai.models.document import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[Document]: ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]: ...

    def can_load(self, path: str) -> bool:
        from pathlib import Path
        return Path(path).suffix.lower() in self.supported_extensions
