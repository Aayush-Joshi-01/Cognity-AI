"""OpenAIVisionOCR — multimodal OCR via GPT-4o vision API."""
from cognity_ai.ocr.base import BaseOCR


class OpenAIVisionOCR(BaseOCR):
    """OCR provider that uses OpenAI GPT-4o vision to extract text from images."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    def ocr(self, image) -> str:
        import base64

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIVisionOCR requires 'openai'. "
                "Install it with: pip install openai"
            ) from exc

        img_bytes = self._read_image_bytes(image)
        img_b64 = base64.b64encode(img_bytes).decode()

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract all text from this image. "
                                "Return only the extracted text, no commentary."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    @property
    def supports_multimodal(self) -> bool:
        return True
