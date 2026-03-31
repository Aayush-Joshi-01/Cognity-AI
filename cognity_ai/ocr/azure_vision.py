"""AzureVisionOCR — multimodal OCR via Azure OpenAI GPT-4o vision."""
from cognity_ai.ocr.base import BaseOCR


class AzureVisionOCR(BaseOCR):
    """OCR provider using Azure OpenAI GPT-4o vision endpoint."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str = "gpt-4o",
        api_version: str = "2024-02-01",
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version

    def ocr(self, image) -> str:
        import base64

        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "AzureVisionOCR requires 'openai'. "
                "Install it with: pip install openai"
            ) from exc

        img_bytes = self._read_image_bytes(image)
        img_b64 = base64.b64encode(img_bytes).decode()

        client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        response = client.chat.completions.create(
            model=self.deployment,
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
