"""AnthropicVisionOCR — multimodal OCR via Claude vision API."""
from cognity_ai.ocr.base import BaseOCR


class AnthropicVisionOCR(BaseOCR):
    """OCR provider that uses Anthropic Claude vision to extract text from images."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key
        self.model = model

    def ocr(self, image) -> str:
        import base64

        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicVisionOCR requires 'anthropic'. "
                "Install it with: pip install anthropic"
            ) from exc

        img_bytes = self._read_image_bytes(image)
        img_b64 = base64.b64encode(img_bytes).decode()

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
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
        )
        return response.content[0].text.strip()

    @property
    def supports_multimodal(self) -> bool:
        return True
