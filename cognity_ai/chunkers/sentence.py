"""SentenceChunker — sentence-boundary chunking (DEFAULT), ported from NLPProcessor.semantic_chunk."""
from cognity_ai.chunkers.base import BaseChunker
from cognity_ai.models.retrieval import SemanticChunk, PageInfo


class SentenceChunker(BaseChunker):
    """Sentence-boundary-aware chunker with entity overlap tracking.

    Each chunk knows which entities it contains, enabling graph-vector linking.
    This is an exact port of NLPProcessor.semantic_chunk() made standalone.

    Requires a loaded spaCy Language model.
    """

    def __init__(self, nlp_model, chunk_sentences: int = 5, overlap: int = 1):
        """
        Args:
            nlp_model: A loaded spaCy Language object.
            chunk_sentences: Number of sentences per chunk.
            overlap: Number of sentences to repeat at the start of the next chunk.
        """
        self.nlp = nlp_model
        self.chunk_sentences = chunk_sentences
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Split text into sentence-boundary chunks.

        Args:
            text: Full document text.
            doc_id: Document identifier used to build chunk IDs.
            pages: Optional list of PageInfo for page attribution.

        Returns:
            Ordered list of SemanticChunk objects.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        chunk_size = self.chunk_sentences
        overlap = self.overlap

        chunks: list[SemanticChunk] = []
        i = 0
        idx = 0
        while i < len(sentences):
            end = min(i + chunk_size, len(sentences))
            chunk_sents = sentences[i:end]
            chunk_text = " ".join(s.text.strip() for s in chunk_sents)

            # Track which named entities appear in this chunk
            chunk_start = chunk_sents[0].start_char
            chunk_end = chunk_sents[-1].end_char
            ent_names = list({
                e.text.strip().title()
                for e in doc.ents
                if e.start_char >= chunk_start and e.end_char <= chunk_end
            })

            # Map to page info if available
            page_info: PageInfo | None = None
            if pages:
                for p in pages:
                    if p.start_char <= chunk_start < p.end_char:
                        page_info = p
                        break

            chunks.append(SemanticChunk(
                chunk_id=f"{doc_id}__chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                page_info=page_info,
                entity_names=ent_names,
                sentence_count=len(chunk_sents),
                token_estimate=len(chunk_text.split()),
            ))
            idx += 1
            i = end - overlap if overlap > 0 and end < len(sentences) else end

        return chunks
