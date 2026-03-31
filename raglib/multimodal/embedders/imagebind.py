"""
raglib.multimodal.embedders.imagebind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ImageBind-based multimodal embedder using Meta's ImageBind model.

ImageBind learns a single shared embedding space across **six modalities**:
image, text, audio, depth, thermal, and IMU data.  This makes it uniquely
powerful for cross-modal retrieval — e.g. searching images using an audio
query, or vice versa.

This embedder exposes image, text, audio, and video-frame embedding (video
frames are treated as images).

.. warning::
    **Experimental** — ImageBind has a non-standard installation process.
    See the installation instructions below.

Supported modalities
--------------------
* ``"image"``
* ``"text"``
* ``"audio"``
* ``"video"`` (frame-level, delegated to image path)

Installation
------------
ImageBind requires manual installation from the Facebook Research repository::

    # 1. Clone the repo
    git clone https://github.com/facebookresearch/ImageBind
    cd ImageBind
    pip install .

    # 2. Then install raglib's multimodal extras for supporting libs
    pip install raglib[imagebind]

For full instructions see: https://github.com/facebookresearch/ImageBind
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from raglib.multimodal.embedders.base import BaseMultimodalEmbedder, _load_image_bytes

if TYPE_CHECKING:
    import torch


_IMAGEBIND_INSTALL_MSG = (
    "ImageBind requires manual install. "
    "See: https://github.com/facebookresearch/ImageBind\n\n"
    "Quick start:\n"
    "  git clone https://github.com/facebookresearch/ImageBind && "
    "cd ImageBind && pip install .\n"
    "  pip install raglib[imagebind]"
)

_IMAGEBIND_DIM = 1024


def _try_import_imagebind() -> tuple:
    """Lazily import ImageBind's model and data utilities.

    Tries the two most common module layouts produced by different ImageBind
    installation methods.

    Returns
    -------
    tuple
        ``(imagebind_model_module, data_module, torch)``

    Raises
    ------
    ImportError
        If ImageBind cannot be found, with installation instructions.
    """
    try:
        import torch

        try:
            # Standard installation via ``pip install .`` from the repo.
            from imagebind import imagebind_model, data as imagebind_data
        except ImportError:
            # Some packaging variants expose it at the top level.
            import imagebind_model  # type: ignore[no-redef]
            import data as imagebind_data  # type: ignore[no-redef]

        return imagebind_model, imagebind_data, torch

    except ImportError as exc:
        raise ImportError(_IMAGEBIND_INSTALL_MSG) from exc


class ImageBindEmbedder(BaseMultimodalEmbedder):
    """Multimodal embedder backed by Meta's ImageBind model.

    ImageBind embeds images, text, audio, and video frames into a single
    1024-dimensional vector space, enabling cross-modal similarity search
    without any paired training data between modalities.

    .. warning::
        This is the most **experimental** embedder in ``raglib.multimodal``.
        The ImageBind API may change; integration relies on Meta's open-source
        release which is not distributed via PyPI.

    Parameters
    ----------
    device:
        Torch device.  ``"auto"`` selects CUDA if available, else CPU.
    pretrained:
        Whether to load pretrained ImageBind weights (recommended).

    Examples
    --------
    >>> embedder = ImageBindEmbedder()
    >>> img_vec = embedder.embed_image("photo.jpg")
    >>> txt_vec = embedder.embed_text("a dog barking")
    >>> aud_vec = embedder.embed_audio("bark.wav")
    >>> import numpy as np
    >>> # All vectors live in the same space — cross-modal dot products work!
    >>> print(np.dot(aud_vec, img_vec))
    """

    def __init__(
        self,
        device: str = "auto",
        pretrained: bool = True,
    ) -> None:
        self._device_arg = device
        self._pretrained = pretrained

        # Lazy-loaded attributes.
        self._model: Optional[Any] = None
        self._torch_device: Optional["torch.device"] = None

        warnings.warn(
            "ImageBindEmbedder is highly experimental and depends on a manual "
            "ImageBind installation from https://github.com/facebookresearch/ImageBind. "
            "APIs may change without notice.",
            stacklevel=2,
            category=UserWarning,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> "torch.device":
        """Resolve the device argument."""
        _, _, torch = _try_import_imagebind()
        if self._device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._device_arg)

    def _ensure_loaded(self) -> None:
        """Load the ImageBind model on first call (lazy init)."""
        if self._model is not None:
            return

        imagebind_model, _, torch = _try_import_imagebind()

        self._torch_device = self._resolve_device()
        self._model = imagebind_model.imagebind_huge(pretrained=self._pretrained)
        self._model.eval()
        self._model.to(self._torch_device)

    @staticmethod
    def _normalise(vector: "np.ndarray") -> list[float]:
        """L2-normalise a 1-D numpy array."""
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector.tolist()
        return (vector / norm).tolist()

    def _run_inference(self, inputs: dict[str, Any]) -> dict[str, "torch.Tensor"]:
        """Run a forward pass on the model with the given inputs dict.

        Parameters
        ----------
        inputs:
            Dictionary mapping ImageBind modality keys to tensors, as produced
            by the ImageBind ``data`` module helpers.

        Returns
        -------
        dict[str, torch.Tensor]
            Embedding tensors keyed by modality name.
        """
        _, _, torch = _try_import_imagebind()
        # Move all tensors to the correct device.
        inputs = {
            k: v.to(self._torch_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            embeddings = self._model(inputs)
        return embeddings

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Embed an image into the ImageBind unified embedding space.

        Parameters
        ----------
        image:
            File path (``str`` / :class:`pathlib.Path`), URL, or raw bytes.
            The image is written to a temporary file if bytes are supplied,
            because the ImageBind data loader expects a file path.

        Returns
        -------
        list[float]
            Unit-normalised 1024-dimensional vector.
        """
        import tempfile
        import os

        self._ensure_loaded()
        _, imagebind_data, _ = _try_import_imagebind()

        raw_bytes = _load_image_bytes(image)

        # ImageBind's data helpers expect filesystem paths; write to a temp file.
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            inputs = {
                imagebind_data.ModalityType.VISION: imagebind_data.load_and_transform_vision_data(
                    [tmp_path], self._torch_device
                )
            }
            embeddings = self._run_inference(inputs)
            vec = embeddings[imagebind_data.ModalityType.VISION].squeeze(0).cpu().numpy()
        finally:
            os.unlink(tmp_path)

        return self._normalise(vec)

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string into the ImageBind unified embedding space.

        Parameters
        ----------
        text:
            The text to embed.

        Returns
        -------
        list[float]
            Unit-normalised 1024-dimensional vector.
        """
        self._ensure_loaded()
        _, imagebind_data, _ = _try_import_imagebind()

        inputs = {
            imagebind_data.ModalityType.TEXT: imagebind_data.load_and_transform_text(
                [text], self._torch_device
            )
        }
        embeddings = self._run_inference(inputs)
        vec = embeddings[imagebind_data.ModalityType.TEXT].squeeze(0).cpu().numpy()
        return self._normalise(vec)

    def embed_audio(self, audio: Union[str, bytes, Path]) -> list[float]:
        """Embed an audio clip into the ImageBind unified embedding space.

        Parameters
        ----------
        audio:
            File path, URL string, or raw audio bytes (WAV/FLAC/MP3).

        Returns
        -------
        list[float]
            Unit-normalised 1024-dimensional vector, in the same space as
            image and text embeddings.
        """
        import tempfile
        import os

        self._ensure_loaded()
        _, imagebind_data, _ = _try_import_imagebind()

        if isinstance(audio, bytes):
            suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio)
                audio_path = tmp.name
            own_tmp = True
        else:
            audio_path = str(audio)
            own_tmp = False

        try:
            inputs = {
                imagebind_data.ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(
                    [audio_path], self._torch_device
                )
            }
            embeddings = self._run_inference(inputs)
            vec = embeddings[imagebind_data.ModalityType.AUDIO].squeeze(0).cpu().numpy()
        finally:
            if own_tmp:
                os.unlink(audio_path)

        return self._normalise(vec)

    def embed_video_frame(self, frame: Union[str, bytes, Path]) -> list[float]:
        """Embed a video frame — identical to :meth:`embed_image` for ImageBind.

        Parameters
        ----------
        frame:
            File path, URL, or raw bytes of a single video frame.

        Returns
        -------
        list[float]
            Unit-normalised 1024-dimensional vector.
        """
        return self.embed_image(frame)

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
            One unit-normalised 1024-dim vector per input.
        """
        if not images:
            return []

        import tempfile
        import os

        self._ensure_loaded()
        _, imagebind_data, _ = _try_import_imagebind()

        tmp_paths: list[str] = []
        try:
            for img in images:
                raw = _load_image_bytes(img)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(raw)
                    tmp_paths.append(tmp.name)

            inputs = {
                imagebind_data.ModalityType.VISION: imagebind_data.load_and_transform_vision_data(
                    tmp_paths, self._torch_device
                )
            }
            embeddings = self._run_inference(inputs)
            vecs = embeddings[imagebind_data.ModalityType.VISION].cpu().numpy()
        finally:
            for p in tmp_paths:
                os.unlink(p)

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).tolist()

    def embed_batch_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings in a single forward pass.

        Parameters
        ----------
        texts:
            List of text strings.

        Returns
        -------
        list[list[float]]
            One unit-normalised 1024-dim vector per input.
        """
        if not texts:
            return []

        self._ensure_loaded()
        _, imagebind_data, _ = _try_import_imagebind()

        inputs = {
            imagebind_data.ModalityType.TEXT: imagebind_data.load_and_transform_text(
                texts, self._torch_device
            )
        }
        embeddings = self._run_inference(inputs)
        vecs = embeddings[imagebind_data.ModalityType.TEXT].cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).tolist()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Output dimensionality of ImageBind embeddings.

        Returns
        -------
        int
            1024 (fixed for all ImageBind-Huge checkpoints).
        """
        return _IMAGEBIND_DIM

    @property
    def supported_modalities(self) -> list[str]:
        """Modalities supported by ImageBind.

        Returns
        -------
        list[str]
            ``["image", "text", "audio", "video"]``
        """
        return ["image", "text", "audio", "video"]
