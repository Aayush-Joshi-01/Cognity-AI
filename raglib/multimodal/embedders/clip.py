"""
raglib.multimodal.embedders.clip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CLIP-based multimodal embedder using OpenAI's CLIP models via HuggingFace
Transformers.

CLIP (Contrastive Language–Image Pre-training) produces a shared embedding
space for images and text, making it ideal for cross-modal retrieval tasks.

Supported models
----------------
* ``"openai/clip-vit-base-patch32"``  — 512-dimensional (default, fast)
* ``"openai/clip-vit-large-patch14"`` — 768-dimensional (higher quality)

Installation
------------
Install the required optional dependencies with::

    pip install raglib[clip]

which resolves to ``transformers torch Pillow``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from raglib.multimodal.embedders.base import BaseMultimodalEmbedder, _load_image_bytes

if TYPE_CHECKING:
    # Import only for type hints; not executed at runtime unless already loaded.
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, CLIPProcessor

# Mapping from model name fragment to output dimensionality.
_CLIP_DIM_MAP: dict[str, int] = {
    "clip-vit-base-patch32": 512,
    "clip-vit-large-patch14": 768,
}

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def _try_import() -> tuple:
    """Lazily import heavy dependencies and return (torch, CLIPModel, CLIPProcessor, Image).

    Raises
    ------
    ImportError
        If any required package is not installed, with a helpful install hint.
    """
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        return torch, CLIPModel, CLIPProcessor, Image
    except ImportError as exc:
        raise ImportError(
            "CLIPEmbedder requires 'transformers', 'torch', and 'Pillow'. "
            "Install with: pip install raglib[clip]"
        ) from exc


class CLIPEmbedder(BaseMultimodalEmbedder):
    """Multimodal embedder backed by OpenAI CLIP via HuggingFace Transformers.

    Produces a shared vector space for both images and text, enabling
    cross-modal similarity search (e.g. retrieve images with a text query or
    retrieve text captions with an image query).

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Supported:

        * ``"openai/clip-vit-base-patch32"`` (512-dim, default)
        * ``"openai/clip-vit-large-patch14"`` (768-dim)
    device:
        Torch device to use.  ``"auto"`` selects CUDA if available, otherwise
        falls back to CPU.

    Examples
    --------
    >>> embedder = CLIPEmbedder()
    >>> img_vec = embedder.embed_image("photo.jpg")
    >>> txt_vec = embedder.embed_text("a dog running on a beach")
    >>> import numpy as np
    >>> similarity = np.dot(img_vec, txt_vec)  # cosine sim (both unit-normed)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device_arg = device

        # Lazy-loaded attributes – populated on first use via _ensure_loaded().
        self._model: Optional[CLIPModel] = None  # type: ignore[name-defined]
        self._processor: Optional[CLIPProcessor] = None  # type: ignore[name-defined]
        self._torch_device: Optional[torch.device] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> "torch.device":
        """Resolve the torch device string to an actual :class:`torch.device`."""
        torch, *_ = _try_import()
        if self._device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_arg)

    def _ensure_loaded(self) -> None:
        """Load the CLIP model and processor on first call (lazy initialisation)."""
        if self._model is not None:
            return

        torch, CLIPModel, CLIPProcessor, _ = _try_import()

        self._torch_device = self._resolve_device()
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        self._model = CLIPModel.from_pretrained(self._model_name).to(
            self._torch_device
        )
        self._model.eval()

    def _bytes_to_pil(self, raw: bytes) -> "PILImage.Image":
        """Convert raw bytes to a PIL Image in RGB mode."""
        _, _, _, Image = _try_import()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def _normalise(vector: "np.ndarray") -> list[float]:
        """L2-normalise a 1-D numpy array and return as a Python float list."""
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector.tolist()
        return (vector / norm).tolist()

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Embed an image and return a unit-normalised CLIP image vector.

        Parameters
        ----------
        image:
            File path (``str`` / :class:`pathlib.Path`), URL, or raw bytes.

        Returns
        -------
        list[float]
            Unit-normalised dense vector of length :attr:`dimensions`.
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        raw_bytes = _load_image_bytes(image)
        pil_image = self._bytes_to_pil(raw_bytes)

        inputs = self._processor(images=pil_image, return_tensors="pt").to(
            self._torch_device
        )

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        return self._normalise(features.squeeze().cpu().numpy())

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string and return a unit-normalised CLIP text vector.

        Parameters
        ----------
        text:
            The text to embed.

        Returns
        -------
        list[float]
            Unit-normalised dense vector of length :attr:`dimensions`.
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self._torch_device)

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        return self._normalise(features.squeeze().cpu().numpy())

    def embed_batch_images(
        self, images: list[Union[str, bytes, Path]]
    ) -> list[list[float]]:
        """Embed a batch of images in a single forward pass for efficiency.

        Parameters
        ----------
        images:
            List of images (path, URL, or bytes each).

        Returns
        -------
        list[list[float]]
            One unit-normalised vector per input image.
        """
        if not images:
            return []

        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        pil_images = [self._bytes_to_pil(_load_image_bytes(img)) for img in images]
        inputs = self._processor(images=pil_images, return_tensors="pt").to(
            self._torch_device
        )

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        return [self._normalise(row.cpu().numpy()) for row in features]

    def embed_batch_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings in a single forward pass.

        Parameters
        ----------
        texts:
            List of text strings.

        Returns
        -------
        list[list[float]]
            One unit-normalised vector per input text.
        """
        if not texts:
            return []

        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self._torch_device)

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        return [self._normalise(row.cpu().numpy()) for row in features]

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Output dimensionality determined from the model name.

        Returns
        -------
        int
            512 for ``clip-vit-base-patch32``, 768 for
            ``clip-vit-large-patch14``.
        """
        for fragment, dim in _CLIP_DIM_MAP.items():
            if fragment in self._model_name:
                return dim
        # Fall back: load the model and read the projection dimension.
        self._ensure_loaded()
        return self._model.config.projection_dim  # type: ignore[union-attr]

    @property
    def supported_modalities(self) -> list[str]:
        """Modalities supported by CLIP.

        Returns
        -------
        list[str]
            ``["image", "text"]``
        """
        return ["image", "text"]
