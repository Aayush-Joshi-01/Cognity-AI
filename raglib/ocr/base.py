"""Abstract base class for OCR providers."""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseOCR(ABC):
    @abstractmethod
    def ocr(self, image: str | bytes | Path) -> str: ...

    @property
    def supports_multimodal(self) -> bool:
        return False

    def _read_image_bytes(self, image) -> bytes:
        if isinstance(image, (str, Path)):
            return Path(image).read_bytes()
        return image
