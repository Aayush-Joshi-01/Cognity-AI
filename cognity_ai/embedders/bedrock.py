"""BedrockEmbedder — embedder using Amazon Bedrock (Titan Embeddings V2)."""
import json

from cognity_ai.embedders.base import BaseEmbedder


class BedrockEmbedder(BaseEmbedder):
    """Embed texts via Amazon Bedrock bedrock-runtime invoke_model.

    Defaults to Amazon Titan Embeddings V2 which produces 1024-dimensional
    vectors.  Texts are embedded one at a time because the Titan model does
    not support batch requests natively.

    Credentials are picked up from the environment / instance profile when
    access_key_id is left empty; explicit keys are used when provided.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        access_key_id: str = "",
        secret_access_key: str = "",
    ):
        self._region = region
        self._model_id = model_id
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key

    def _get_client(self):
        import boto3

        kwargs: dict = {"region_name": self._region}
        if self._access_key_id:
            kwargs["aws_access_key_id"] = self._access_key_id
            kwargs["aws_secret_access_key"] = self._secret_access_key
        return boto3.client("bedrock-runtime", **kwargs)

    def embed_batch(
        self, texts: list[str], task_type: str = "retrieval_document"
    ) -> list[list[float]]:
        client = self._get_client()
        results: list[list[float]] = []
        for text in texts:
            body = json.dumps({"inputText": text})
            resp = client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            results.append(result["embedding"])
        return results

    def embed_query(self, query: str) -> list[float]:
        return self.embed_batch([query])[0]

    @property
    def dimensions(self) -> int:
        return 1024  # Amazon Titan Embeddings V2 default
