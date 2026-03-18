"""
raglib.multimodal.embedders.siglip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SigLIP-based multimodal embedder using Google's SigLIP models via HuggingFace
Transformers.

SigLIP (Sigmoid Loss for Language-Image Pre-training) improves upon CLIP by
using a sigmoid pairwise loss instead of softmax, leading to better zero-shot
transfer performance, especially at smaller batch sizes.

Supported models
----------------
* ``"google/siglip-base-patch16-224"``  — 768-dimensional (default)
* ``"google/siglip-large-patch16-256"`` — 1024-dimensional (higher quality)

Installation
------------
Install the required optional dependencies with::

    pip install raglib[siglip]

which resolves to ``transformers torch Pillow``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from raglib.multimodal.embedders.base import BaseMultimodalEmbedder, _load_image_bytes

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModel, AutoProcessor

_DEFAULT_MODEL = "google/siglip-base-patch16-224"


def _try_import() -> tuple:
    """Lazily import heavy dependencies; return (torch, AutoModel, AutoProcessor, Image).

    Raises
    ------
    ImportError
        If any required package is missing, with a helpful install hint.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        return torch, AutoModel, AutoProcessor, Image
    except ImportError as exc:
        raise ImportError(
            "SigLIPEmbedder requires 'transformers', 'torch', and 'Pillow'. "
            "Install with: pip install raglib[siglip]"
        ) from exc


class SigLIPEmbedder(BaseMultimodalEmbedder):
    """Multimodal embedder backed by Google SigLIP via HuggingFace Transformers.

    SigLIP produces a shared embedding space for images and text, similar to
    CLIP but with improved training objectives.  Mean-pooling is applied to
    the patch / token outputs to obtain fixed-size vectors.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Supported:

        * ``"google/siglip-base-patch16-224"`` (768-dim, default)
        * ``"google/siglip-large-patch16-256"`` (1024-dim)
    device:
        Torch device.  ``"auto"`` picks CUDA if available, else CPU.

    Examples
    --------
    >>> embedder = SigLIPEmbedder()
    >>> img_vec = embedder.embed_image("photo.jpg")
    >>> txt_vec = embedder.embed_text("a cat sitting on a mat")
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

        # Lazy-loaded attributes.
        self._model: Optional[AutoModel] = None  # type: ignore[name-defined]
        self._processor: Optional[AutoProcessor] = None  # type: ignore[name-defined]
        self._torch_device: Optional[torch.device] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> "torch.device":
        """Resolve the device argument to a :class:`torch.device`."""
        torch, *_ = _try_import()
        if self._device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_arg)

    def _ensure_loaded(self) -> None:
        """Load the SigLIP model and processor on first call (lazy init)."""
        if self._model is not None:
            return

        torch, AutoModel, AutoProcessor, _ = _try_import()

        self._torch_device = self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(
            self._torch_device
        )
        self._model.eval()

    def _bytes_to_pil(self, raw: bytes) -> "PILImage.Image":
        """Convert raw bytes to a PIL Image in RGB mode."""
        _, _, _, Image = _try_import()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def _mean_pool_and_normalise(hidden_states: "torch.Tensor") -> list[float]:
        """Mean-pool along the sequence dimension, then L2-normalise.

        Parameters
        ----------
        hidden_states:
            Tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns
        -------
        list[float]
            Unit-normalised 1-D vector (Python float list).
        """
        # Mean over sequence / patch dimension → (batch, hidden_dim)
        pooled = hidden_states.mean(dim=1).squeeze(0).detach().cpu().numpy()
        norm = np.linalg.norm(pooled)
        if norm == 0.0:
            return pooled.tolist()
        return (pooled / norm).tolist()

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Embed an image and return a unit-normalised SigLIP vision vector.

        Parameters
        ----------
        image:
            File path, URL string, or raw bytes.

        Returns
        -------
        list[float]
            Unit-normalised vector of length :attr:`dimensions`.
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        raw_bytes = _load_image_bytes(image)
        pil_image = self._bytes_to_pil(raw_bytes)

        inputs = self._processor(images=pil_image, return_tensors="pt").to(
            self._torch_device
        )

        with torch.no_grad():
            outputs = self._model.vision_model(**inputs)

        return self._mean_pool_and_normalise(outputs.last_hidden_state)

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string and return a unit-normalised SigLIP text vector.

        Parameters
        ----------
        text:
            The text to embed.

        Returns
        -------
        list[float]
            Unit-normalised vector of length :attr:`dimensions`.
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        inputs = self._processor(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        ).to(self._torch_device)

        with torch.no_grad():
            outputs = self._model.text_model(**inputs)

        return self._mean_pool_and_normalise(outputs.last_hidden_state)

    def embed_batch_images(
        self, images: list[Union[str, bytes, Path]]
    ) -> list[list[float]]:
        """Embed a batch of images in a single forward pass.

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
            outputs = self._model.vision_model(**inputs)

        # outputs.last_hidden_state: (batch, seq, hidden)
        pooled = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = pooled / norms
        return normalised.tolist()

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
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        ).to(self._torch_device)

        with torch.no_grad():
            outputs = self._model.text_model(**inputs)

        pooled = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = pooled / norms
        return normalised.tolist()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Output dimensionality of the SigLIP model.

        Attempts to read the hidden size from the loaded model config.  Falls
        back to well-known defaults when the model has not yet been loaded.

        Returns
        -------
        int
            768 for ``siglip-base-patch16-224``, 1024 for
            ``siglip-large-patch16-256``.
        """
        if self._model is not None:
            try:
                return self._model.config.vision_config.hidden_size
            except AttributeError:
                pass

        # Known defaults before the model is loaded.
        if "large" in self._model_name:
            return 1024
        return 768

    @property
    def supported_modalities(self) -> list[str]:
        """Modalities supported by SigLIP.

        Returns
        -------
        list[str]
            ``["image", "text"]``
        """
        return ["image", "text"]
