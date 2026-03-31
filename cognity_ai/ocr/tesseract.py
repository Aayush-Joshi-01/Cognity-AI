"""TesseractOCR — traditional OCR via pytesseract + Pillow."""
from cognity_ai.ocr.base import BaseOCR


class TesseractOCR(BaseOCR):
    """OCR provider backed by Tesseract via pytesseract."""

    def __init__(self, lang: str = "eng", config: str = ""):
        self.lang = lang
        self.config = config

    def ocr(self, image) -> str:
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise ImportError(
                "TesseractOCR requires 'pytesseract' and 'Pillow'. "
                "Install them with: pip install pytesseract Pillow"
            ) from exc

        img_bytes = self._read_image_bytes(image)

        from io import BytesIO
        pil_image = Image.open(BytesIO(img_bytes))
        return pytesseract.image_to_string(pil_image, lang=self.lang, config=self.config)

    @property
    def supports_multimodal(self) -> bool:
        return False
