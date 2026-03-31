"""
raglib.multimodal.retrievers.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class for all multimodal retrievers in the
``raglib.multimodal`` extension.

All concrete multimodal retrievers must subclass
:class:`BaseMultimodalRetriever` and implement at minimum:

* :meth:`retrieve` — text query → list of :class:`MultimodalRetrievalResult`
* :attr:`modality` — a string identifying the modality handled

Optional overrides:

* :meth:`retrieve_by_image` — image query → similar content
* :meth:`retrieve_by_audio` — audio query → similar content

.. note::
    This module is part of the experimental ``raglib.multimodal`` extension.
    APIs may change without notice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from raglib.multimodal.models.media import MultimodalRetrievalResult


class BaseMultimodalRetriever(ABC):
    """Abstract base class for multimodal retrieval strategies.

    Subclasses handle a single modality (image, video, audio) or a
    cross-modal combination and expose a uniform interface for text-driven
    as well as cross-modal queries.

    All retrievers must implement :meth:`retrieve` and :attr:`modality`.
    The remaining methods have sensible default implementations that can be
    overridden for richer behaviour.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve the most relevant multimodal chunks for a text query.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of retrieval results, most relevant first.
        """

    @property
    @abstractmethod
    def modality(self) -> str:
        """Modality identifier for this retriever.

        Returns
        -------
        str
            One of ``"image"``, ``"video"``, ``"audio"``, or
            ``"cross_modal"``.
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def retrieve_by_image(
        self,
        image: Union[str, bytes, Path],
        top_k: int = 5,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve content similar to a query image.

        The default implementation raises :exc:`NotImplementedError`.
        Override in subclasses that support image-to-X retrieval (e.g.
        :class:`ImageRetriever`, :class:`CrossModalRetriever`).

        Parameters
        ----------
        image:
            Query image as a filesystem path (``str`` or
            :class:`pathlib.Path`) or raw image bytes.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of retrieval results, most relevant first.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support image-based retrieval. "
            "Use ImageRetriever or CrossModalRetriever for image queries."
        )

    def retrieve_by_audio(
        self,
        audio: Union[str, bytes, Path],
        top_k: int = 5,
    ) -> list[MultimodalRetrievalResult]:
        """Retrieve content similar to a query audio clip.

        The default implementation raises :exc:`NotImplementedError`.
        Override in subclasses that support audio-to-X retrieval (e.g.
        :class:`AudioRetriever`, :class:`CrossModalRetriever`).

        Parameters
        ----------
        audio:
            Query audio as a filesystem path (``str`` or
            :class:`pathlib.Path`) or raw audio bytes.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[MultimodalRetrievalResult]
            Ranked list of retrieval results, most relevant first.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support audio-based retrieval. "
            "Use AudioRetriever or CrossModalRetriever for audio queries."
        )

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> str:
        """Retrieve results and return a formatted plain-text description.

        The default implementation concatenates the ``text`` field of each
        result.  Subclasses with access to a generator should override this
        to produce fluent, grounded answers.

        Parameters
        ----------
        question:
            Natural-language question to answer.
        top_k:
            Maximum number of results to retrieve before formatting.

        Returns
        -------
        str
            Human-readable summary of the retrieved multimodal content.
        """
        results = self.retrieve(question, top_k=top_k)
        parts: list[str] = []
        for result in results:
            label = f"[{result.modality.upper()}]"
            text = result.text or "(no text description)"
            parts.append(f"{label} {text}")
        return "\n\n".join(parts) if parts else "No relevant results found."

    def query_with_sources(
        self, question: str, top_k: int = 5
    ) -> dict[str, object]:
        """Retrieve results and return both the answer and source metadata.

        Parameters
        ----------
        question:
            Natural-language question to answer.
        top_k:
            Maximum number of results to retrieve.

        Returns
        -------
        dict
            A dictionary with two keys:

            ``"answer"`` : str
                Formatted answer produced by :meth:`query`.
            ``"sources"`` : list[MultimodalRetrievalResult]
                The raw retrieval results used to build the answer.
        """
        results = self.retrieve(question, top_k=top_k)
        answer = self.query(question, top_k=top_k)
        return {"answer": answer, "sources": results}

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(modality={self.modality!r})"
