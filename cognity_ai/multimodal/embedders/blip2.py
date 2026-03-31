"""
raglib.multimodal.embedders.blip2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Salesforce BLIP-2 embedder/captioner.

BLIP-2 (Bootstrapping Language-Image Pre-training with Frozen Image Encoders
and Large Language Models) is primarily an **image→text** model designed for:

* Automatic image captioning
* Visual question answering (VQA)
* Image-conditioned text generation

It is **not** a joint-embedding model in the same sense as CLIP or SigLIP —
there is no native text encoder that projects text into the same space as
images.  However, the Q-Former (query transformer) produces 256-dimensional
vision query vectors that can serve as image embeddings for image-to-image
retrieval.

.. note::
    For joint text+image embeddings, prefer
    :class:`~raglib.multimodal.embedders.clip.CLIPEmbedder` or
    :class:`~raglib.multimodal.embedders.siglip.SigLIPEmbedder`.

Supported models
----------------
* ``"Salesforce/blip2-opt-2.7b"`` (default — requires ~6 GB VRAM)
* ``"Salesforce/blip2-opt-6.7b"``
* ``"Salesforce/blip2-flan-t5-xl"``

Installation
------------
Install the required optional dependencies with::

    pip install raglib[blip2]

which resolves to ``transformers torch Pillow accelerate``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from cognity_ai.multimodal.embedders.base import BaseMultimodalEmbedder, _load_image_bytes

if TYPE_CHECKING:
    import torch
    from PIL import Image as PILImage
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

_DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"
_BLIP2_QUERY_DIM = 256  # Q-Former output dimensionality


def _try_import() -> tuple:
    """Lazily import heavy BLIP-2 dependencies.

    Returns
    -------
    tuple
        ``(torch, Blip2Processor, Blip2ForConditionalGeneration, Image)``

    Raises
    ------
    ImportError
        If any required package is missing, with a helpful install hint.
    """
    try:
        import torch
        from PIL import Image
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        return torch, Blip2Processor, Blip2ForConditionalGeneration, Image
    except ImportError as exc:
        raise ImportError(
            "BLIP2Embedder requires 'transformers', 'torch', 'Pillow', and 'accelerate'. "
            "Install with: pip install raglib[blip2]"
        ) from exc


class BLIP2Embedder(BaseMultimodalEmbedder):
    """Image captioner and visual question answerer backed by Salesforce BLIP-2.

    Primary capabilities:

    * :meth:`caption_image` — generates a natural-language caption for an image.
    * :meth:`answer_question` — answers a question about an image (VQA).
    * :meth:`embed_image` — extracts a 256-dim vision query vector from the
      Q-Former (useful for image-to-image similarity; **not** comparable to
      text embeddings).

    .. note::
        :meth:`embed_text` is intentionally **not supported**.  BLIP-2 has no
        text encoder that produces embeddings in the same space as its image
        embeddings.  Use :class:`~raglib.multimodal.embedders.clip.CLIPEmbedder`
        or :class:`~raglib.multimodal.embedders.siglip.SigLIPEmbedder` for
        joint text–image retrieval.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for a BLIP-2 checkpoint.
    device:
        Torch device.  ``"auto"`` picks CUDA if available, else CPU.

    Examples
    --------
    >>> embedder = BLIP2Embedder()
    >>> caption = embedder.caption_image("photo.jpg")
    >>> print(caption)
    'a golden retriever playing fetch on a sunny beach'

    >>> answer = embedder.answer_question("photo.jpg", "What colour is the dog?")
    >>> print(answer)
    'golden'

    >>> img_vec = embedder.embed_image("photo.jpg")  # 256-dim Q-Former vector
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device_arg = device

        # Lazy-loaded attributes.
        self._model: Optional[Blip2ForConditionalGeneration] = None  # type: ignore[name-defined]
        self._processor: Optional[Blip2Processor] = None  # type: ignore[name-defined]
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
        """Load the BLIP-2 model and processor on first call (lazy init).

        BLIP-2 models are large; loading is deferred until the first actual
        use to avoid penalising imports.
        """
        if self._model is not None:
            return

        torch, Blip2Processor, Blip2ForConditionalGeneration, _ = _try_import()

        self._torch_device = self._resolve_device()
        self._processor = Blip2Processor.from_pretrained(self._model_name)

        # ``device_map="auto"`` is preferred on CUDA for multi-GPU / 8-bit
        # quantisation support via accelerate.
        if self._torch_device.type == "cuda":
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self._model_name, device_map="auto", torch_dtype=torch.float16
            )
        else:
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self._model_name
            ).to(self._torch_device)

        self._model.eval()

    def _bytes_to_pil(self, raw: bytes) -> "PILImage.Image":
        """Convert raw bytes to a PIL Image in RGB mode."""
        _, _, _, Image = _try_import()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def _normalise(vector: "np.ndarray") -> list[float]:
        """L2-normalise a 1-D numpy array."""
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector.tolist()
        return (vector / norm).tolist()

    # ------------------------------------------------------------------
    # Primary capabilities: captioning and VQA
    # ------------------------------------------------------------------

    def caption_image(self, image: Union[str, bytes, Path]) -> str:
        """Generate a natural-language caption describing an image.

        Parameters
        ----------
        image:
            File path, URL string, or raw image bytes.

        Returns
        -------
        str
            Generated caption string (decoded, stripped of prompt artefacts).
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        raw_bytes = _load_image_bytes(image)
        pil_image = self._bytes_to_pil(raw_bytes)

        inputs = self._processor(images=pil_image, return_tensors="pt").to(
            self._torch_device
        )

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=50)

        caption: str = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return caption

    def answer_question(
        self, image: Union[str, bytes, Path], question: str
    ) -> str:
        """Answer a free-form question about an image (visual question answering).

        Parameters
        ----------
        image:
            File path, URL string, or raw image bytes.
        question:
            Natural-language question about the image content.

        Returns
        -------
        str
            Model-generated answer string.
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        raw_bytes = _load_image_bytes(image)
        pil_image = self._bytes_to_pil(raw_bytes)

        inputs = self._processor(
            images=pil_image, text=question, return_tensors="pt"
        ).to(self._torch_device)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=30)

        answer: str = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return answer

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Extract a 256-dim vision query embedding from the BLIP-2 Q-Former.

        Uses the Q-Former's output query tokens from the vision encoder —
        specifically the CLS-like first query token — as a compact image
        representation.  This embedding is suitable for image-to-image
        similarity but is **not** cross-modally comparable to text embeddings.

        Parameters
        ----------
        image:
            File path, URL string, or raw bytes.

        Returns
        -------
        list[float]
            Unit-normalised 256-dimensional vector.
        """
        self._ensure_loaded()
        torch, _, _, _ = _try_import()

        raw_bytes = _load_image_bytes(image)
        pil_image = self._bytes_to_pil(raw_bytes)

        inputs = self._processor(images=pil_image, return_tensors="pt").to(
            self._torch_device
        )

        with torch.no_grad():
            # Run the vision encoder + Q-Former without the language model.
            vision_outputs = self._model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            image_embeds = vision_outputs.last_hidden_state  # (1, patches, hidden)

            query_tokens = self._model.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )
            query_outputs = self._model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
            )
            # query_outputs.last_hidden_state: (1, num_queries, qformer_hidden)
            # Use the first query token as the embedding (CLS-like).
            vec = query_outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        return self._normalise(vec)

    def embed_text(self, text: str) -> list[float]:
        """Not supported by BLIP-2.

        BLIP-2 is an image→text model and does not produce text embeddings in
        the same vector space as image embeddings.

        Raises
        ------
        NotImplementedError
            Always.  Use :class:`~raglib.multimodal.embedders.clip.CLIPEmbedder`
            or :class:`~raglib.multimodal.embedders.siglip.SigLIPEmbedder`
            for joint text–image embeddings.
        """
        raise NotImplementedError(
            "BLIP-2 is primarily an image→text model; use CLIPEmbedder or "
            "SigLIPEmbedder for joint text-image embeddings."
        )

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Output dimensionality of the BLIP-2 Q-Former query embeddings.

        Returns
        -------
        int
            256 (fixed Q-Former query transformer output dimension).
        """
        return _BLIP2_QUERY_DIM

    @property
    def supported_modalities(self) -> list[str]:
        """Modalities supported by this BLIP-2 embedder.

        Returns
        -------
        list[str]
            ``["image"]`` — text embedding is not supported.
        """
        return ["image"]
