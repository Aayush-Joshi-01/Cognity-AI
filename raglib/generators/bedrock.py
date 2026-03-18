"""BedrockGenerator — generator using Amazon Bedrock (Claude models via Messages API)."""
import json

from raglib.generators.base import BaseGenerator


class BedrockGenerator(BaseGenerator):
    """Generate answers via Amazon Bedrock bedrock-runtime invoke_model.

    Defaults to Anthropic Claude 3.5 Sonnet v2 accessed through the Bedrock
    Converse/invoke_model interface using the bedrock-2023-05-31 anthropic_version.

    Credentials are picked up from the environment / instance profile when
    access_key_id is left empty; explicit keys are used when provided.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        access_key_id: str = "",
        secret_access_key: str = "",
    ):
        self._region = region
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key

    def _get_client(self):
        import boto3

        kwargs: dict = {"region_name": self._region}
        if self._access_key_id:
            kwargs["aws_access_key_id"] = self._access_key_id
            kwargs["aws_secret_access_key"] = self._secret_access_key
        return boto3.client("bedrock-runtime", **kwargs)

    def generate(self, question: str, context: str) -> str:
        client = self._get_client()

        if question:
            user_content = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            # Pre-built prompt passed via generate_with_structured_context
            user_content = context

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self._max_tokens,
                "system": "Use the provided context to answer accurately.",
                "messages": [{"role": "user", "content": user_content}],
                "temperature": self._temperature,
            }
        )
        resp = client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"]
