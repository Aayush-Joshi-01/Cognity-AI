"""GeminiVisionOCR — multimodal OCR via Google Gemini 2.0 Flash."""
from raglib.ocr.base import BaseOCR


class GeminiVisionOCR(BaseOCR):
    """OCR provider that uses Gemini vision to extract text from images."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model

    def ocr(self, image) -> str:
        try:
            import google.generativeai as genai
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "GeminiVisionOCR requires 'google-generativeai' and 'Pillow'. "
                "Install them with: pip install google-generativeai Pillow"
            ) from exc

        from io import BytesIO

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        img_bytes = self._read_image_bytes(image)
        pil_image = Image.open(BytesIO(img_bytes))

        prompt = (
            "Extract all text from this image. "
            "Return only the extracted text, no commentary."
        )
        response = model.generate_content([prompt, pil_image])
        return response.text.strip()

    @property
    def supports_multimodal(self) -> bool:
        return True
