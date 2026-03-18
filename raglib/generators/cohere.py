"""CohereGenerator — generator using the Cohere Chat API."""
from raglib.generators.base import BaseGenerator


class CohereGenerator(BaseGenerator):
    """Generate answers using Cohere's chat endpoint (command-r-plus or similar).

    The cohere client is created lazily to avoid importing the package at
    module load time.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-plus",
        temperature: float = 0.1,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def generate(self, question: str, context: str) -> str:
        import cohere

        client = cohere.Client(self._api_key)

        if question:
            message = question
            preamble = f"Use this context to answer: {context}"
        else:
            # Pre-built prompt passed via generate_with_structured_context;
            # treat it as the message with no separate preamble.
            message = context
            preamble = ""

        kwargs = {
            "model": self._model,
            "message": message,
            "temperature": self._temperature,
        }
        if preamble:
            kwargs["preamble"] = preamble

        resp = client.chat(**kwargs)
        return resp.text
