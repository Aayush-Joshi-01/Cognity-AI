"""GeminiVisionOCR — multimodal OCR via Google Gemini 2.0 Flash (google-genai SDK)."""
import os
from cognity_ai.ocr.base import BaseOCR


class GeminiVisionOCR(BaseOCR):
    """OCR provider that uses Gemini vision to extract text from images.

    Accepts a ``GeminiConfig`` object, an explicit ``api_key`` string, or no
    arguments — in which case ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` is read
    from the environment.  Optionally use ``project_id`` + ``use_vertexai=True``
    for project-based Vertex AI access.
    """

    def __init__(
        self,
        config=None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        project_id: str | None = None,
        location: str = "us-central1",
        use_vertexai: bool = False,
        timeout: int = 120,
    ):
        # Config object
        if config is not None and not isinstance(config, str):
            api_key = api_key or getattr(config, "api_key", "") or None
            model = model or getattr(config, "model", "gemini-2.0-flash")
            project_id = project_id or getattr(config, "project_id", "") or None
            location = getattr(config, "location", location)
            use_vertexai = use_vertexai or getattr(config, "use_vertexai", False)
            timeout = getattr(config, "timeout", timeout)
        elif isinstance(config, str):
            api_key = api_key or config

        self.model_name = model or "gemini-2.0-flash"
        self._api_key = api_key
        self._project_id = project_id
        self._location = location
        self._use_vertexai = use_vertexai
        self._timeout = timeout

    def _get_client(self):
        try:
            from google import genai
            from google.genai import types as gentypes
        except ImportError as exc:
            raise ImportError(
                "GeminiVisionOCR requires 'google-genai' and 'Pillow'. "
                "Install them with: pip install google-genai Pillow"
            ) from exc

        http_opts = gentypes.HttpOptions(timeout=self._timeout)
        if self._use_vertexai and self._project_id:
            return genai.Client(
                vertexai=True,
                project=self._project_id,
                location=self._location,
                http_options=http_opts,
            )
        if self._api_key:
            return genai.Client(api_key=self._api_key, http_options=http_opts)
        return genai.Client(http_options=http_opts)

    def ocr(self, image) -> str:
        try:
            from google.genai import types as gentypes
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "GeminiVisionOCR requires 'google-genai' and 'Pillow'. "
                "Install them with: pip install google-genai Pillow"
            ) from exc

        from io import BytesIO

        client = self._get_client()
        img_bytes = self._read_image_bytes(image)

        # Detect format via PIL and convert to bytes for the API
        pil_image = Image.open(BytesIO(img_bytes))
        fmt = (pil_image.format or "JPEG").upper()
        buf = BytesIO()
        pil_image.save(buf, format=fmt)
        img_data = buf.getvalue()
        mime_type = f"image/{fmt.lower()}"

        prompt = (
            "Extract all text from this image. "
            "Return only the extracted text, no commentary."
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                prompt,
                gentypes.Part.from_bytes(data=img_data, mime_type=mime_type),
            ],
        )
        return response.text.strip()

    @property
    def supports_multimodal(self) -> bool:
        return True
