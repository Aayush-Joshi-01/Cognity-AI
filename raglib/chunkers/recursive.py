"""RecursiveChunker — LangChain-style recursive character splitting."""
from raglib.chunkers.base import BaseChunker
from raglib.models.retrieval import SemanticChunk, PageInfo


class RecursiveChunker(BaseChunker):
    """Splits text recursively through a priority list of separators.

    Tries each separator in order: [\"\\n\\n\", \"\\n\", \". \", \" \", \"\"].
    For each separator, if a piece is still too large, it recurses with the
    next separator until all pieces are within chunk_size characters.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: list | None = None,
    ):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of characters carried over into the next chunk.
            separators: Ordered list of separator strings to try.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]

    # ── Public API ───────────────────────────────────────────────────────

    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Recursively split text and return SemanticChunk objects.

        Args:
            text: Full document text.
            doc_id: Document identifier used to build chunk IDs.
            pages: Optional list of PageInfo for page attribution.

        Returns:
            Ordered list of SemanticChunk objects.
        """
        raw_chunks = self._split_recursive(text, self.separators)

        result: list[SemanticChunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            # Approximate start char by scanning for chunk_text in original text
            start_char = text.find(chunk_text)

            page_info: PageInfo | None = None
            if pages and start_char >= 0:
                for p in pages:
                    if p.start_char <= start_char < p.end_char:
                        page_info = p
                        break

            result.append(SemanticChunk(
                chunk_id=f"{doc_id}__chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                page_info=page_info,
                entity_names=[],
                sentence_count=0,
                token_estimate=len(chunk_text.split()),
            ))

        # Re-number sequentially after dropping empty chunks
        for new_idx, sc in enumerate(result):
            sc.index = new_idx
            sc.chunk_id = f"{doc_id}__chunk_{new_idx}"

        return result

    # ── Core splitting logic ─────────────────────────────────────────────

    def _split_recursive(self, text: str, separators: list) -> list:
        """Return a list of text pieces that are each ≤ chunk_size characters."""
        if not text:
            return []

        # If text already fits, return as-is
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep == "":
                # Hard character split as last resort
                return self._hard_split(text)

            parts = text.split(sep)
            if len(parts) <= 1:
                continue  # Separator not found, try next

            # Merge small parts back together until we hit the size limit
            merged: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        merged.append(current)
                    # If the single part is still too large, recurse with remaining seps
                    if len(part) > self.chunk_size:
                        merged.extend(self._split_recursive(part, separators[i + 1:]))
                        current = ""
                    else:
                        current = part

            if current:
                merged.append(current)

            # Apply overlap: carry the tail of each chunk into the start of the next
            if self.overlap > 0:
                merged = self._apply_overlap(merged)

            return merged

        return [text]

    def _hard_split(self, text: str) -> list:
        """Split at exact character boundaries with overlap."""
        pieces = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            pieces.append(text[start:end])
            start += self.chunk_size - self.overlap
            if start >= len(text):
                break
        return pieces

    def _apply_overlap(self, chunks: list) -> list:
        """Prepend the tail of each chunk (up to overlap chars) onto the next chunk."""
        if len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[i - 1][-self.overlap:] if self.overlap > 0 else ""
            overlapped.append(tail + chunks[i])
        return overlapped
