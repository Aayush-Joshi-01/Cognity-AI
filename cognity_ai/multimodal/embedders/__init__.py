"""
raglib.multimodal.embedders
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multimodal embedding backends.

All heavy model dependencies are lazily imported so that importing this
package does not require every optional backend to be installed.  Only
:class:`~raglib.multimodal.embedders.base.BaseMultimodalEmbedder` is loaded
eagerly.

Lazy-loaded embedders
---------------------
- :class:`CLIPEmbedder`      — ``raglib.multimodal.embedders.clip``
- :class:`SigLIPEmbedder`    — ``raglib.multimodal.embedders.siglip``
- :class:`ImageBindEmbedder` — ``raglib.multimodal.embedders.imagebind``
- :class:`BLIP2Embedder`     — ``raglib.multimodal.embedders.blip2``
"""

from cognity_ai.multimodal.embedders.base import BaseMultimodalEmbedder

__all__ = ["BaseMultimodalEmbedder"]

_LAZY_MAP: dict[str, str] = {
    "CLIPEmbedder": "raglib.multimodal.embedders.clip",
    "SigLIPEmbedder": "raglib.multimodal.embedders.siglip",
    "ImageBindEmbedder": "raglib.multimodal.embedders.imagebind",
    "BLIP2Embedder": "raglib.multimodal.embedders.blip2",
}


def __getattr__(name: str):  # noqa: ANN001, ANN202
    if name in _LAZY_MAP:
        import importlib

        mod = importlib.import_module(_LAZY_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
