"""LLM generation providers — import the one you need or use the factory."""

from raglib.generators.base import BaseGenerator

# Concrete implementations are imported lazily to avoid hard dependency errors.
# Use `from raglib.generators.gemini import GeminiGenerator` directly, or
# let the factory wire the correct generator via LibraryConfig.llm.

__all__ = ["BaseGenerator"]


def __getattr__(name):  # noqa: PLC0103
    """Allow `from raglib.generators import GeminiGenerator` without eager imports."""
    _map = {
        "GeminiGenerator": ("raglib.generators.gemini", "GeminiGenerator"),
        "VertexAIGenerator": ("raglib.generators.vertex_ai", "VertexAIGenerator"),
        "OpenAIGenerator": ("raglib.generators.openai", "OpenAIGenerator"),
        "AzureOpenAIGenerator": ("raglib.generators.azure_openai", "AzureOpenAIGenerator"),
        "AnthropicGenerator": ("raglib.generators.anthropic", "AnthropicGenerator"),
        "BedrockGenerator": ("raglib.generators.bedrock", "BedrockGenerator"),
        "CohereGenerator": ("raglib.generators.cohere", "CohereGenerator"),
        "OllamaGenerator": ("raglib.generators.ollama", "OllamaGenerator"),
    }
    if name in _map:
        module_path, class_name = _map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'raglib.generators' has no attribute {name!r}")
