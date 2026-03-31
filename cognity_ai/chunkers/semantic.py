"""SemanticChunker — groups consecutive sentences by embedding cosine similarity."""
import re

from cognity_ai.chunkers.base import BaseChunker
from cognity_ai.models.retrieval import SemanticChunk, PageInfo


class SemanticChunker(BaseChunker):
    """Groups consecutive sentences into chunks when their embeddings are similar.

    Requires an embedder that exposes an ``embed_batch(texts) -> list[list[float]]``
    method (or compatible interface).  A spaCy model can optionally be supplied
    for more accurate sentence segmentation; otherwise a simple regex fallback
    is used.
    """

    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.8,
        max_sentences: int = 10,
        nlp_model=None,
    ):
        """
        Args:
            embedder: Object with ``embed_batch(texts: list[str]) -> list[list[float]]``.
            similarity_threshold: Cosine similarity below which a new chunk starts.
            max_sentences: Hard cap on sentences per chunk regardless of similarity.
            nlp_model: Optional spaCy Language for sentence splitting.
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_sentences = max_sentences
        self.nlp = nlp_model

    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Split text into semantically coherent chunks.

        Args:
            text: Full document text.
            doc_id: Document identifier used to build chunk IDs.
            pages: Optional list of PageInfo for page attribution.

        Returns:
            Ordered list of SemanticChunk objects.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Embed all sentences in one batch call
        embeddings = self.embedder.embed_batch(sentences)

        # Group consecutive sentences by similarity
        groups: list[list[int]] = []  # each group is a list of sentence indices
        current_group: list[int] = [0]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if (
                sim >= self.similarity_threshold
                and len(current_group) < self.max_sentences
            ):
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        groups.append(current_group)

        # Build SemanticChunk objects
        chunks: list[SemanticChunk] = []
        for idx, group in enumerate(groups):
            chunk_text = " ".join(sentences[i] for i in group)
            start_char = text.find(sentences[group[0]])

            page_info: PageInfo | None = None
            if pages and start_char >= 0:
                for p in pages:
                    if p.start_char <= start_char < p.end_char:
                        page_info = p
                        break

            chunks.append(SemanticChunk(
                chunk_id=f"{doc_id}__chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                page_info=page_info,
                entity_names=[],
                sentence_count=len(group),
                token_estimate=len(chunk_text.split()),
            ))

        return chunks

    # ── Helpers ──────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list:
        """Return a list of sentence strings."""
        if self.nlp is not None:
            doc = self.nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        # Regex fallback: split on sentence-ending punctuation
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
