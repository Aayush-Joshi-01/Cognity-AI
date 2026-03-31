"""
raglib.multimodal.embedders.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for all multimodal embedders in the ``raglib.multimodal``
extension.

All concrete embedders must subclass :class:`BaseMultimodalEmbedder` and
implement at minimum :meth:`embed_image`, :meth:`embed_text`,
:attr:`dimensions`, and :attr:`supported_modalities`.

Helper
------
:func:`_load_image_bytes` — Normalise an image argument (path, URL, or raw
bytes) into raw bytes ready for further processing.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_image_bytes(image: Union[str, bytes, Path]) -> bytes:
    """Load an image from a file path, ``Path`` object, or raw bytes.

    Parameters
    ----------
    image:
        One of:

        * ``str`` — filesystem path or URL string.  File paths are read from
          disk; URL strings (starting with ``http://`` or ``https://``) are
          fetched with :mod:`urllib.request`.
        * :class:`pathlib.Path` — read from disk.
        * ``bytes`` — returned as-is.

    Returns
    -------
    bytes
        Raw image bytes.

    Raises
    ------
    FileNotFoundError
        If a filesystem path is given but the file does not exist.
    ValueError
        If the *image* argument is of an unsupported type.
    """
    if isinstance(image, bytes):
        return image

    if isinstance(image, Path):
        return image.read_bytes()

    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(image) as response:  # noqa: S310
                return response.read()

        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image!r}")
        return path.read_bytes()

    raise ValueError(
        f"Unsupported image type {type(image).__name__!r}. "
        "Expected str (path/URL), bytes, or pathlib.Path."
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMultimodalEmbedder(ABC):
    """Abstract base class for multimodal embedding models.

    Concrete subclasses must implement:

    * :meth:`embed_image`
    * :meth:`embed_text`
    * :attr:`dimensions`
    * :attr:`supported_modalities`

    Optional overrides:

    * :meth:`embed_audio` — raises :exc:`NotImplementedError` by default.
    * :meth:`embed_video_frame` — delegates to :meth:`embed_image` by default.
    * :meth:`embed_batch_images` — iterates :meth:`embed_image` by default.
    * :meth:`embed_batch_texts` — iterates :meth:`embed_text` by default.
    """

    # ------------------------------------------------------------------
    # Abstract single-item methods
    # ------------------------------------------------------------------

    @abstractmethod
    def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Embed a single image and return a unit-normalised float vector.

        Parameters
        ----------
        image:
            File path (``str`` or :class:`pathlib.Path`), a URL string, or
            raw image bytes.

        Returns
        -------
        list[float]
            Unit-normalised dense embedding vector of length :attr:`dimensions`.
        """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string and return a unit-normalised float vector.

        Parameters
        ----------
        text:
            The text string to embed.

        Returns
        -------
        list[float]
            Unit-normalised dense embedding vector of length :attr:`dimensions`.
        """

    # ------------------------------------------------------------------
    # Optional single-item overrides
    # ------------------------------------------------------------------

    def embed_audio(self, audio: Union[str, bytes, Path]) -> list[float]:
        """Embed a single audio clip.

        Not supported by most embedders.  Override in subclasses that have
        native audio support (e.g. :class:`~raglib.multimodal.embedders.imagebind.ImageBindEmbedder`).

        Parameters
        ----------
        audio:
            File path, URL string, or raw audio bytes.

        Raises
        ------
        NotImplementedError
            Always, unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support audio embedding. "
            "Use ImageBindEmbedder for cross-modal audio embeddings."
        )

    def embed_video_frame(self, frame: Union[str, bytes, Path]) -> list[float]:
        """Embed a single video frame image.

        By default delegates to :meth:`embed_image`.  Override if the
        embedder requires special handling for video frames.

        Parameters
        ----------
        frame:
            File path, URL string, or raw image bytes of the frame.

        Returns
        -------
        list[float]
            Unit-normalised dense embedding vector of length :attr:`dimensions`.
        """
        return self.embed_image(frame)

    # ------------------------------------------------------------------
    # Batch helpers (default: iterate single-item methods)
    # ------------------------------------------------------------------

    def embed_batch_images(
        self, images: list[Union[str, bytes, Path]]
    ) -> list[list[float]]:
        """Embed a batch of images.

        The default implementation calls :meth:`embed_image` sequentially.
        Override for batched GPU inference to avoid per-item overhead.

        Parameters
        ----------
        images:
            List of images (each as path, URL, or bytes).

        Returns
        -------
        list[list[float]]
            One embedding vector per input image.
        """
        return [self.embed_image(img) for img in images]

    def embed_batch_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        The default implementation calls :meth:`embed_text` sequentially.
        Override for batched GPU inference.

        Parameters
        ----------
        texts:
            List of text strings.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        return [self.embed_text(t) for t in texts]

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output dimensionality of the embedding vectors produced by this model.

        Returns
        -------
        int
            Number of floating-point values in each output embedding.
        """

    @property
    @abstractmethod
    def supported_modalities(self) -> list[str]:
        """List of modality names this embedder supports.

        Typical values include ``"image"``, ``"text"``, ``"audio"``,
        ``"video"``.

        Returns
        -------
        list[str]
            Supported modality names.
        """

    # ------------------------------------------------------------------
    # Concrete property
    # ------------------------------------------------------------------

    @property
    def modality(self) -> str:
        """Human-readable identifier for this embedder class.

        Returns the class name by default (e.g. ``"CLIPEmbedder"``).

        Returns
        -------
        str
            Class name of this embedder instance.
        """
        return type(self).__name__

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"dimensions={self.dimensions}, "
            f"modalities={self.supported_modalities!r})"
        )
