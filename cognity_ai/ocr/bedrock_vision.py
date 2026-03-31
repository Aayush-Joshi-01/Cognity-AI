"""BedrockVisionOCR — multimodal OCR via AWS Bedrock Claude."""
from cognity_ai.ocr.base import BaseOCR


class BedrockVisionOCR(BaseOCR):
    """OCR provider that uses AWS Bedrock Claude multimodal API."""

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        access_key_id: str = "",
        secret_access_key: str = "",
    ):
        self.region = region
        self.model_id = model_id
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key

    def ocr(self, image) -> str:
        import base64
        import json

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "BedrockVisionOCR requires 'boto3'. "
                "Install it with: pip install boto3"
            ) from exc

        img_bytes = self._read_image_bytes(image)
        img_b64 = base64.b64encode(img_bytes).decode()

        session_kwargs: dict = {"region_name": self.region}
        if self.access_key_id and self.secret_access_key:
            session_kwargs["aws_access_key_id"] = self.access_key_id
            session_kwargs["aws_secret_access_key"] = self.secret_access_key

        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock-runtime")

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
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
            }
        )

        response = client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()

    @property
    def supports_multimodal(self) -> bool:
        return True
