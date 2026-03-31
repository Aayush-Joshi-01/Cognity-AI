"""HybridChunker — sentence boundaries first, optional semantic regrouping."""
from cognity_ai.chunkers.base import BaseChunker
from cognity_ai.chunkers.sentence import SentenceChunker
from cognity_ai.models.retrieval import SemanticChunk, PageInfo


class HybridChunker(BaseChunker):
    """Two-phase chunker: sentence splitting followed by optional semantic merging.

    Phase 1 (always): Run SentenceChunker to get sentence-boundary-aligned chunks.
    Phase 2 (optional): When ``use_semantic_regroup=True`` and an embedder is
        provided, merge adjacent chunks whose embeddings exceed the similarity
        threshold.
    """

    def __init__(
        self,
        nlp_model,
        embedder=None,
        chunk_sentences: int = 5,
        overlap: int = 1,
        use_semantic_regroup: bool = False,
        similarity_threshold: float = 0.85,
    ):
        """
        Args:
            nlp_model: Loaded spaCy Language object (required for SentenceChunker).
            embedder: Optional embedder with ``embed_batch(texts) -> list[list[float]]``.
            chunk_sentences: Number of sentences per initial chunk.
            overlap: Sentence overlap between consecutive chunks.
            use_semantic_regroup: Whether to merge adjacent similar chunks.
            similarity_threshold: Cosine similarity threshold for merging.
        """
        self.sentence_chunker = SentenceChunker(
            nlp_model=nlp_model,
            chunk_sentences=chunk_sentences,
            overlap=overlap,
        )
        self.embedder = embedder
        self.use_semantic_regroup = use_semantic_regroup
        self.similarity_threshold = similarity_threshold

    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Split text into hybrid chunks.

        Args:
            text: Full document text.
            doc_id: Document identifier.
            pages: Optional list of PageInfo for page attribution.

        Returns:
            Ordered list of SemanticChunk objects.
        """
        # Phase 1: sentence-boundary splitting
        initial_chunks = self.sentence_chunker.chunk(text, doc_id, pages)

        if not initial_chunks:
            return initial_chunks

        # Phase 2: optional semantic regrouping
        if self.use_semantic_regroup and self.embedder is not None:
            return self._semantic_regroup(initial_chunks, doc_id)

        return initial_chunks

    # ── Semantic regrouping ──────────────────────────────────────────────

    def _semantic_regroup(
        self,
        chunks: list[SemanticChunk],
        doc_id: str,
    ) -> list[SemanticChunk]:
        """Merge adjacent chunks where cosine similarity exceeds the threshold."""
        if len(chunks) <= 1:
            return chunks

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        merged: list[SemanticChunk] = []
        current = chunks[0]
        current_embedding = embeddings[0]

        for i in range(1, len(chunks)):
            sim = self._cosine_similarity(current_embedding, embeddings[i])
            if sim >= self.similarity_threshold:
                # Merge chunk i into current
                merged_text = current.text + " " + chunks[i].text
                merged_entity_names = list(
                    set(current.entity_names + chunks[i].entity_names)
                )
                current = SemanticChunk(
                    chunk_id=current.chunk_id,  # keep the earlier chunk's id
                    doc_id=doc_id,
                    text=merged_text,
                    index=current.index,
                    page_info=current.page_info,
                    entity_names=merged_entity_names,
                    sentence_count=current.sentence_count + chunks[i].sentence_count,
                    token_estimate=len(merged_text.split()),
                )
                # Update embedding as average of the two
                current_embedding = [
                    (a + b) / 2.0
                    for a, b in zip(current_embedding, embeddings[i])
                ]
            else:
                merged.append(current)
                current = chunks[i]
                current_embedding = embeddings[i]

        merged.append(current)

        # Re-number sequentially
        for new_idx, sc in enumerate(merged):
            sc.index = new_idx
            sc.chunk_id = f"{doc_id}__chunk_{new_idx}"

        return merged

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
