"""VertexAIGenerator — generator using Google Cloud Vertex AI Generative Models."""
from raglib.generators.base import BaseGenerator


class VertexAIGenerator(BaseGenerator):
    """Generate answers using a Vertex AI GenerativeModel (e.g. gemini-1.5-pro).

    Vertex AI SDK objects are created lazily so the package is only imported
    when the generator is actually used.
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
    ):
        self._project = project
        self._location = location
        self._model = model
        self._temperature = temperature

    def generate(self, question: str, context: str) -> str:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self._project, location=self._location)
        model = GenerativeModel(self._model)
        if question:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # Pre-built prompt passed via generate_with_structured_context
            prompt = context
        response = model.generate_content(
            prompt,
            generation_config={"temperature": self._temperature},
        )
        return response.text
