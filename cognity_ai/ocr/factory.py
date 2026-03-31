"""OCRFactory — instantiate OCR providers by name from config."""
from cognity_ai.ocr.base import BaseOCR


class OCRFactory:
    """Factory for creating OCR provider instances."""

    @staticmethod
    def create(provider: str, config=None) -> BaseOCR:
        """Create an OCR provider instance.

        Args:
            provider: One of "gemini_vision", "openai_vision", "anthropic_vision",
                      "azure_vision", "bedrock_vision", "tesseract".
            config: A LibraryConfig (or compatible) object whose provider sub-configs
                    are used for credentials/settings. Pass None to use defaults.

        Returns:
            A BaseOCR instance.
        """
        provider = provider.lower().strip()

        if provider == "gemini_vision":
            from cognity_ai.ocr.gemini_vision import GeminiVisionOCR
            api_key = config.gemini.api_key if config else ""
            model = config.gemini.model if config else "gemini-2.0-flash"
            return GeminiVisionOCR(api_key=api_key, model=model)

        if provider == "openai_vision":
            from cognity_ai.ocr.openai_vision import OpenAIVisionOCR
            api_key = config.openai.api_key if config else ""
            model = config.openai.model if config else "gpt-4o"
            return OpenAIVisionOCR(api_key=api_key, model=model)

        if provider == "anthropic_vision":
            from cognity_ai.ocr.anthropic_vision import AnthropicVisionOCR
            api_key = config.anthropic.api_key if config else ""
            model = config.anthropic.model if config else "claude-sonnet-4-6"
            return AnthropicVisionOCR(api_key=api_key, model=model)

        if provider == "azure_vision":
            from cognity_ai.ocr.azure_vision import AzureVisionOCR
            if config:
                return AzureVisionOCR(
                    endpoint=config.azure_openai.endpoint,
                    api_key=config.azure_openai.api_key,
                    deployment=config.azure_openai.deployment_name,
                    api_version=config.azure_openai.api_version,
                )
            return AzureVisionOCR(endpoint="", api_key="")

        if provider == "bedrock_vision":
            from cognity_ai.ocr.bedrock_vision import BedrockVisionOCR
            if config:
                return BedrockVisionOCR(
                    region=config.bedrock.region,
                    model_id=config.bedrock.model_id,
                    access_key_id=config.bedrock.access_key_id,
                    secret_access_key=config.bedrock.secret_access_key,
                )
            return BedrockVisionOCR()

        if provider == "tesseract":
            from cognity_ai.ocr.tesseract import TesseractOCR
            return TesseractOCR()

        raise ValueError(
            f"Unknown OCR provider: '{provider}'. "
            "Valid options: gemini_vision, openai_vision, anthropic_vision, "
            "azure_vision, bedrock_vision, tesseract."
        )

    @staticmethod
    def create_with_fallback(providers: list, config=None) -> BaseOCR:
        """Try each provider in order, return the first one that instantiates successfully.

        Args:
            providers: Ordered list of provider names to try.
            config: Configuration object passed to each provider.

        Returns:
            First successfully created BaseOCR instance.

        Raises:
            RuntimeError: If no provider could be instantiated.
        """
        last_error: Exception | None = None
        for p in providers:
            try:
                return OCRFactory.create(p, config)
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"No OCR provider available from list {providers}. "
            f"Last error: {last_error}"
        )
