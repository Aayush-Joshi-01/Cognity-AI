"""ParentChildChunker — two-level chunking for retrieval + context."""
from cognity_ai.chunkers.base import BaseChunker
from cognity_ai.models.retrieval import SemanticChunk, PageInfo


class ParentChildChunker(BaseChunker):
    """Creates parent chunks (for context) and child chunks (for retrieval).

    Child chunks are the primary retrieval units and have ``parent_chunk_id`` set.
    Parent chunks have ``is_parent=True`` and provide broader context.
    """

    def __init__(
        self,
        child_size: int = 200,
        parent_size: int = 800,
        child_overlap: int = 20,
        parent_overlap: int = 50,
    ):
        """
        Args:
            child_size: Number of words per child chunk.
            parent_size: Number of words per parent chunk.
            child_overlap: Word overlap between consecutive child chunks.
            parent_overlap: Word overlap between consecutive parent chunks.
        """
        self.child_size = child_size
        self.parent_size = parent_size
        self.child_overlap = child_overlap
        self.parent_overlap = parent_overlap

    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Return both parent and child chunks interleaved.

        Children come first (primary retrieval units), then parents.
        Children reference their parent via ``parent_chunk_id``.

        Args:
            text: Full document text.
            doc_id: Document identifier.
            pages: Optional list of PageInfo for page attribution.

        Returns:
            List containing child chunks followed by parent chunks.
        """
        words = text.split(" ")

        # Build parent chunks
        parents = self._build_word_chunks(
            words=words,
            doc_id=doc_id,
            chunk_size=self.parent_size,
            overlap=self.parent_overlap,
            prefix="parent",
            is_parent=True,
            pages=pages,
        )

        # Build child chunks within each parent's word range
        all_children: list[SemanticChunk] = []
        child_idx = 0

        for parent in parents:
            parent_words = parent.text.split(" ")
            children = self._build_word_chunks(
                words=parent_words,
                doc_id=doc_id,
                chunk_size=self.child_size,
                overlap=self.child_overlap,
                prefix=f"child_{parent.chunk_id}",
                is_parent=False,
                pages=pages,
                start_index=child_idx,
                parent_chunk_id=parent.chunk_id,
            )
            all_children.extend(children)
            child_idx += len(children)

        # Re-number children sequentially
        for new_idx, child in enumerate(all_children):
            child.index = new_idx
            child.chunk_id = f"{doc_id}__chunk_{new_idx}"

        # Return children first (retrieval units), then parents (context)
        return all_children + parents

    # ── Internal helper ──────────────────────────────────────────────────

    def _build_word_chunks(
        self,
        words: list,
        doc_id: str,
        chunk_size: int,
        overlap: int,
        prefix: str,
        is_parent: bool,
        pages: list | None,
        start_index: int = 0,
        parent_chunk_id: str | None = None,
    ) -> list[SemanticChunk]:
        chunks: list[SemanticChunk] = []
        idx = start_index
        i = 0
        step = max(chunk_size - overlap, 1)

        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = " ".join(chunk_words)

            # Approximate start character
            prefix_chars = len(" ".join(words[:i]))
            if i > 0:
                prefix_chars += 1  # the space before position i

            page_info: PageInfo | None = None
            if pages:
                for p in pages:
                    if p.start_char <= prefix_chars < p.end_char:
                        page_info = p
                        break

            chunks.append(SemanticChunk(
                chunk_id=f"{doc_id}__{prefix}_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                page_info=page_info,
                entity_names=[],
                sentence_count=0,
                token_estimate=len(chunk_words),
                is_parent=is_parent,
                parent_chunk_id=parent_chunk_id,
            ))
            idx += 1
            i += step

        return chunks
