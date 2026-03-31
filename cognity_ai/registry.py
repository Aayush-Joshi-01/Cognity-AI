"""
PluginRegistry — central store for all pluggable component classes.
Each subsystem registers its built-in implementations here.
Users can register custom implementations via RAGLibrary.register_*().
"""
from __future__ import annotations
from typing import Type


class PluginRegistry:
    """Central registry for all raglib plugin types."""

    _loaders: dict[str, Type] = {}         # ext → BaseLoader subclass
    _chunkers: dict[str, Type] = {}         # name → BaseChunker subclass
    _ocr: dict[str, Type] = {}              # name → BaseOCR subclass
    _page_indexes: dict[str, Type] = {}     # name → BasePageIndex subclass
    _extractors: dict[str, Type] = {}       # name → BaseExtractor subclass
    _embedders: dict[str, Type] = {}        # name → BaseEmbedder subclass
    _vector_stores: dict[str, Type] = {}    # name → BaseVectorStore subclass
    _graph_stores: dict[str, Type] = {}     # name → BaseGraphStore subclass
    _generators: dict[str, Type] = {}       # name → BaseGenerator subclass
    _retrievers: dict[str, Type] = {}       # name → BaseRetriever subclass

    # ── Loaders ──────────────────────────────────────────────────────────

    @classmethod
    def register_loader(cls, ext: str, klass: Type):
        """Register a loader for a file extension (e.g. '.pdf')."""
        cls._loaders[ext.lower()] = klass

    @classmethod
    def get_loader(cls, ext: str) -> Type | None:
        return cls._loaders.get(ext.lower())

    @classmethod
    def list_loaders(cls) -> list[str]:
        return list(cls._loaders.keys())

    # ── Chunkers ─────────────────────────────────────────────────────────

    @classmethod
    def register_chunker(cls, name: str, klass: Type):
        cls._chunkers[name] = klass

    @classmethod
    def get_chunker(cls, name: str) -> Type:
        if name not in cls._chunkers:
            raise KeyError(f"Chunker '{name}' not registered. Available: {list(cls._chunkers)}")
        return cls._chunkers[name]

    # ── OCR ──────────────────────────────────────────────────────────────

    @classmethod
    def register_ocr(cls, name: str, klass: Type):
        cls._ocr[name] = klass

    @classmethod
    def get_ocr(cls, name: str) -> Type:
        if name not in cls._ocr:
            raise KeyError(f"OCR provider '{name}' not registered. Available: {list(cls._ocr)}")
        return cls._ocr[name]

    # ── Page Indexes ─────────────────────────────────────────────────────

    @classmethod
    def register_page_index(cls, name: str, klass: Type):
        cls._page_indexes[name] = klass

    @classmethod
    def get_page_index(cls, name: str) -> Type:
        if name not in cls._page_indexes:
            raise KeyError(f"PageIndex '{name}' not registered. Available: {list(cls._page_indexes)}")
        return cls._page_indexes[name]

    # ── Extractors ───────────────────────────────────────────────────────

    @classmethod
    def register_extractor(cls, name: str, klass: Type):
        cls._extractors[name] = klass

    @classmethod
    def get_extractor(cls, name: str) -> Type:
        if name not in cls._extractors:
            raise KeyError(f"Extractor '{name}' not registered. Available: {list(cls._extractors)}")
        return cls._extractors[name]

    # ── Embedders ────────────────────────────────────────────────────────

    @classmethod
    def register_embedder(cls, name: str, klass: Type):
        cls._embedders[name] = klass

    @classmethod
    def get_embedder(cls, name: str) -> Type:
        if name not in cls._embedders:
            raise KeyError(f"Embedder '{name}' not registered. Available: {list(cls._embedders)}")
        return cls._embedders[name]

    # ── Vector Stores ────────────────────────────────────────────────────

    @classmethod
    def register_vector_store(cls, name: str, klass: Type):
        cls._vector_stores[name] = klass

    @classmethod
    def get_vector_store(cls, name: str) -> Type:
        if name not in cls._vector_stores:
            raise KeyError(f"VectorStore '{name}' not registered. Available: {list(cls._vector_stores)}")
        return cls._vector_stores[name]

    # ── Graph Stores ─────────────────────────────────────────────────────

    @classmethod
    def register_graph_store(cls, name: str, klass: Type):
        cls._graph_stores[name] = klass

    @classmethod
    def get_graph_store(cls, name: str) -> Type | None:
        return cls._graph_stores.get(name)

    # ── Generators ───────────────────────────────────────────────────────

    @classmethod
    def register_generator(cls, name: str, klass: Type):
        cls._generators[name] = klass

    @classmethod
    def get_generator(cls, name: str) -> Type:
        if name not in cls._generators:
            raise KeyError(f"Generator '{name}' not registered. Available: {list(cls._generators)}")
        return cls._generators[name]

    # ── Retrievers ───────────────────────────────────────────────────────

    @classmethod
    def register_retriever(cls, name: str, klass: Type):
        cls._retrievers[name] = klass

    @classmethod
    def get_retriever(cls, name: str) -> Type:
        if name not in cls._retrievers:
            raise KeyError(f"Retriever '{name}' not registered. Available: {list(cls._retrievers)}")
        return cls._retrievers[name]

    @classmethod
    def summary(cls) -> dict:
        return {
            "loaders": list(cls._loaders.keys()),
            "chunkers": list(cls._chunkers.keys()),
            "ocr": list(cls._ocr.keys()),
            "page_indexes": list(cls._page_indexes.keys()),
            "extractors": list(cls._extractors.keys()),
            "embedders": list(cls._embedders.keys()),
            "vector_stores": list(cls._vector_stores.keys()),
            "graph_stores": list(cls._graph_stores.keys()),
            "generators": list(cls._generators.keys()),
            "retrievers": list(cls._retrievers.keys()),
        }
