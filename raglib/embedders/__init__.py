"""Embedding providers — import the one you need or use the factory."""

from raglib.embedders.base import BaseEmbedder

# Concrete implementations are imported lazily to avoid hard dependency errors.
# Use `from raglib.embedders.gemini import GeminiEmbedder` directly, or
# let the factory wire the correct embedder via LibraryConfig.embedder.

__all__ = ["BaseEmbedder"]


def __getattr__(name):  # noqa: PLC0103
    """Allow `from raglib.embedders import GeminiEmbedder` without eager imports."""
    _map = {
        "GeminiEmbedder": ("raglib.embedders.gemini", "GeminiEmbedder"),
        "VertexAIEmbedder": ("raglib.embedders.vertex_ai", "VertexAIEmbedder"),
        "OpenAIEmbedder": ("raglib.embedders.openai", "OpenAIEmbedder"),
        "AzureOpenAIEmbedder": ("raglib.embedders.azure_openai", "AzureOpenAIEmbedder"),
        "BedrockEmbedder": ("raglib.embedders.bedrock", "BedrockEmbedder"),
        "CohereEmbedder": ("raglib.embedders.cohere", "CohereEmbedder"),
        "SentenceTransformerEmbedder": (
            "raglib.embedders.sentence_transformers", "SentenceTransformerEmbedder",
        ),
        "OllamaEmbedder": ("raglib.embedders.ollama", "OllamaEmbedder"),
    }
    if name in _map:
        module_path, class_name = _map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'raglib.embedders' has no attribute {name!r}")
