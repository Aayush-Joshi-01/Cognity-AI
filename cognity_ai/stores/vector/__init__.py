"""Vector store backends — import the one you need or use the factory."""

from cognity_ai.stores.vector.base import BaseVectorStore

# Concrete implementations are imported lazily to avoid hard dependency errors.
# Use `from cognity_ai.stores.vector.chroma import ChromaStore` directly, or
# let the factory wire the correct store via LibraryConfig.vector_store.

__all__ = ["BaseVectorStore"]


def __getattr__(name):  # noqa: PLC0103
    """Allow `from cognity_ai.stores.vector import ChromaStore` without eager imports."""
    _map = {
        "ChromaStore": ("raglib.stores.vector.chroma", "ChromaStore"),
        "QdrantStore": ("raglib.stores.vector.qdrant", "QdrantStore"),
        "PineconeStore": ("raglib.stores.vector.pinecone", "PineconeStore"),
        "FAISSStore": ("raglib.stores.vector.faiss", "FAISSStore"),
        "WeaviateStore": ("raglib.stores.vector.weaviate", "WeaviateStore"),
        "MilvusStore": ("raglib.stores.vector.milvus", "MilvusStore"),
        "PgVectorStore": ("raglib.stores.vector.pgvector", "PgVectorStore"),
        "AzureSearchStore": ("raglib.stores.vector.azure_search", "AzureSearchStore"),
    }
    if name in _map:
        module_path, class_name = _map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'raglib.stores.vector' has no attribute {name!r}")
