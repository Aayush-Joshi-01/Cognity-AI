"""
raglib.multimodal.stores
~~~~~~~~~~~~~~~~~~~~~~~~

Vector store backends for multimodal RAG.

:class:`~raglib.multimodal.stores.base.BaseMultimodalStore` is loaded eagerly;
concrete implementations are lazy-loaded to avoid hard dependency errors at
import time.

Lazy-loaded stores
------------------
- :class:`ChromaMultimodalStore` — ``raglib.multimodal.stores.chroma_mm``
- :class:`QdrantMultimodalStore` — ``raglib.multimodal.stores.qdrant_mm``

.. warning::
    This subpackage is **experimental**. APIs may change without notice.
"""

from cognity_ai.multimodal.stores.base import BaseMultimodalStore

__all__ = ["BaseMultimodalStore"]

_LAZY_MAP = {
    "ChromaMultimodalStore": "raglib.multimodal.stores.chroma_mm",
    "QdrantMultimodalStore": "raglib.multimodal.stores.qdrant_mm",
}


def __getattr__(name: str):  # noqa: ANN001, ANN202
    if name in _LAZY_MAP:
        import importlib

        mod = importlib.import_module(_LAZY_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
