"""FixedChunker — word-count-based fixed-size chunks with overlap."""
from raglib.chunkers.base import BaseChunker
from raglib.models.retrieval import SemanticChunk, PageInfo


class FixedChunker(BaseChunker):
    """Splits text into chunks of a fixed number of words with a sliding overlap.

    No spaCy dependency — works on any text.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separator: str = " ",
    ):
        """
        Args:
            chunk_size: Number of words per chunk.
            overlap: Number of words to repeat at the start of the next chunk.
            separator: Token separator used when splitting and re-joining.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Split text into fixed-size word chunks.

        Args:
            text: Full document text.
            doc_id: Document identifier used to build chunk IDs.
            pages: Optional list of PageInfo for page attribution.

        Returns:
            Ordered list of SemanticChunk objects.
        """
        words = text.split(self.separator)
        chunks: list[SemanticChunk] = []
        idx = 0
        i = 0

        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = self.separator.join(chunk_words)

            # Approximate character offsets for page attribution
            prefix_chars = len(self.separator.join(words[:i]))
            if i > 0:
                prefix_chars += len(self.separator)  # account for the separator before word i
            chunk_start_char = prefix_chars

            # Map to page info if available
            page_info: PageInfo | None = None
            if pages:
                for p in pages:
                    if p.start_char <= chunk_start_char < p.end_char:
                        page_info = p
                        break
                    # Also catch the case where the chunk starts before the first page boundary
                    if chunk_start_char < p.start_char and not page_info:
                        page_info = p

            chunks.append(SemanticChunk(
                chunk_id=f"{doc_id}__chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                page_info=page_info,
                entity_names=[],
                sentence_count=0,
                token_estimate=len(chunk_words),
            ))
            idx += 1

            step = self.chunk_size - self.overlap
            if step <= 0:
                step = 1
            i += step

        return chunks
