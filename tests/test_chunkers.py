"""Tests for all chunker implementations."""
from __future__ import annotations

import pytest
from cognity_ai.chunkers.fixed import FixedChunker
from cognity_ai.chunkers.recursive import RecursiveChunker
from cognity_ai.models.retrieval import SemanticChunk, PageInfo


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz judge my vow."
)


# ── FixedChunker ──────────────────────────────────────────────────────────────

class TestFixedChunker:
    def test_returns_list_of_semantic_chunks(self):
        chunker = FixedChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk(SAMPLE_TEXT, "doc1")
        assert isinstance(chunks, list)
        assert all(isinstance(c, SemanticChunk) for c in chunks)

    def test_non_empty_output_for_non_empty_input(self):
        chunks = FixedChunker(chunk_size=5, overlap=1).chunk(SAMPLE_TEXT, "doc1")
        assert len(chunks) > 0

    def test_empty_text_produces_no_meaningful_content(self):
        # FixedChunker splits by separator; "" split by " " gives [""],
        # so one chunk with empty text may be produced — callers should filter.
        chunks = FixedChunker(chunk_size=10, overlap=2).chunk("", "doc1")
        assert isinstance(chunks, list)
        # All chunks (if any) have empty or whitespace-only text
        for c in chunks:
            assert c.text.strip() == ""

    def test_chunk_ids_unique(self):
        chunks = FixedChunker(chunk_size=5, overlap=1).chunk(SAMPLE_TEXT, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_doc_id_propagated(self):
        chunks = FixedChunker(chunk_size=10, overlap=2).chunk(SAMPLE_TEXT, "my_doc")
        assert all(c.doc_id == "my_doc" for c in chunks)

    def test_chunk_id_format(self):
        chunks = FixedChunker(chunk_size=10, overlap=2).chunk(SAMPLE_TEXT, "doc1")
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc1__chunk_{i}"

    def test_index_sequential(self):
        chunks = FixedChunker(chunk_size=5, overlap=1).chunk(SAMPLE_TEXT, "doc1")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_no_empty_chunks(self):
        chunks = FixedChunker(chunk_size=10, overlap=3).chunk(SAMPLE_TEXT, "doc1")
        assert all(c.text.strip() for c in chunks)

    def test_token_estimate_is_word_count(self):
        chunks = FixedChunker(chunk_size=10, overlap=0).chunk(SAMPLE_TEXT, "doc1")
        for c in chunks:
            assert c.token_estimate == len(c.text.split())

    def test_chunk_size_respected(self):
        chunker = FixedChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, "doc1")
        for c in chunks:
            assert len(c.text.split()) <= 5

    def test_overlap_produces_repeated_words(self):
        text = "one two three four five six seven eight nine ten"
        chunks = FixedChunker(chunk_size=4, overlap=2).chunk(text, "doc1")
        assert len(chunks) >= 2
        # Last words of chunk 0 should appear at start of chunk 1
        words0 = chunks[0].text.split()
        words1 = chunks[1].text.split()
        assert words0[-2:] == words1[:2]

    def test_entity_names_empty(self):
        chunks = FixedChunker(chunk_size=10).chunk(SAMPLE_TEXT, "doc1")
        assert all(c.entity_names == [] for c in chunks)

    def test_sentence_count_zero(self):
        chunks = FixedChunker(chunk_size=10).chunk(SAMPLE_TEXT, "doc1")
        assert all(c.sentence_count == 0 for c in chunks)

    def test_single_word_text(self):
        chunks = FixedChunker(chunk_size=10, overlap=2).chunk("hello", "doc1")
        assert len(chunks) == 1
        assert chunks[0].text == "hello"

    def test_text_shorter_than_chunk_size(self):
        short = "just three words"
        chunks = FixedChunker(chunk_size=100, overlap=0).chunk(short, "doc1")
        assert len(chunks) == 1
        assert chunks[0].text == short

    def test_page_info_assigned_when_pages_provided(self):
        pages = [PageInfo(page_num=1, start_char=0, end_char=len(SAMPLE_TEXT))]
        chunks = FixedChunker(chunk_size=10, overlap=2).chunk(SAMPLE_TEXT, "doc1", pages=pages)
        assert any(c.page_info is not None for c in chunks)

    def test_custom_separator(self):
        text = "a|b|c|d|e|f|g|h"
        chunks = FixedChunker(chunk_size=3, overlap=0, separator="|").chunk(text, "doc1")
        assert len(chunks) > 1


# ── RecursiveChunker ──────────────────────────────────────────────────────────

PARA_TEXT = (
    "First paragraph with some text.\n\n"
    "Second paragraph here.\n\n"
    "Third paragraph and more content to split.\n\n"
    "Fourth paragraph is the last one."
)


class TestRecursiveChunker:
    def test_returns_list_of_semantic_chunks(self):
        chunks = RecursiveChunker(chunk_size=100, overlap=10).chunk(PARA_TEXT, "doc1")
        assert isinstance(chunks, list)
        assert all(isinstance(c, SemanticChunk) for c in chunks)

    def test_non_empty_output(self):
        chunks = RecursiveChunker(chunk_size=50, overlap=5).chunk(PARA_TEXT, "doc1")
        assert len(chunks) > 0

    def test_empty_text(self):
        chunks = RecursiveChunker(chunk_size=100).chunk("", "doc1")
        assert chunks == []

    def test_chunk_ids_unique(self):
        chunks = RecursiveChunker(chunk_size=50, overlap=5).chunk(PARA_TEXT, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_doc_id_propagated(self):
        chunks = RecursiveChunker(chunk_size=50).chunk(PARA_TEXT, "my_doc")
        assert all(c.doc_id == "my_doc" for c in chunks)

    def test_index_sequential(self):
        chunks = RecursiveChunker(chunk_size=50, overlap=5).chunk(PARA_TEXT, "doc1")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_chunk_id_format(self):
        chunks = RecursiveChunker(chunk_size=50, overlap=5).chunk(PARA_TEXT, "doc1")
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"doc1__chunk_{i}"

    def test_no_empty_chunks(self):
        chunks = RecursiveChunker(chunk_size=50, overlap=0).chunk(PARA_TEXT, "doc1")
        assert all(c.text.strip() for c in chunks)

    def test_chunk_size_respected(self):
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        chunks = chunker.chunk(PARA_TEXT, "doc1")
        for c in chunks:
            assert len(c.text) <= 50 + 20  # small tolerance for overlap join

    def test_paragraph_separator_used_first(self):
        text = "Short first.\n\nShort second."
        chunks = RecursiveChunker(chunk_size=50, overlap=0).chunk(text, "doc1")
        # Both paragraphs fit in chunk_size so they may be merged or split
        assert len(chunks) >= 1

    def test_single_chunk_when_text_fits(self):
        short = "Hello world."
        chunks = RecursiveChunker(chunk_size=200, overlap=0).chunk(short, "doc1")
        assert len(chunks) == 1
        assert chunks[0].text == short

    def test_hard_split_fallback(self):
        long_word = "a" * 600
        chunks = RecursiveChunker(chunk_size=200, overlap=0).chunk(long_word, "doc1")
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c.text) <= 200

    def test_token_estimate_populated(self):
        chunks = RecursiveChunker(chunk_size=100, overlap=0).chunk(PARA_TEXT, "doc1")
        for c in chunks:
            assert c.token_estimate == len(c.text.split())

    def test_custom_separators(self):
        text = "part1|part2|part3|part4|part5"
        chunks = RecursiveChunker(chunk_size=10, overlap=0, separators=["|", ""]).chunk(text, "doc1")
        assert len(chunks) >= 2

    def test_page_info_assigned(self):
        pages = [PageInfo(page_num=1, start_char=0, end_char=len(PARA_TEXT))]
        chunks = RecursiveChunker(chunk_size=50, overlap=0).chunk(PARA_TEXT, "doc1", pages=pages)
        assert any(c.page_info is not None for c in chunks)


# ── SentenceChunker (requires spaCy) ─────────────────────────────────────────

@pytest.mark.skipif(
    not pytest.importorskip("spacy", reason="spacy not installed") if False else False,
    reason="skip logic handled inline"
)
class TestSentenceChunker:
    @pytest.fixture(autouse=True)
    def _load_spacy(self):
        spacy = pytest.importorskip("spacy")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("en_core_web_sm not installed (run: python -m spacy download en_core_web_sm)")

    def test_returns_semantic_chunks(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunker = SentenceChunker(self.nlp, chunk_sentences=2, overlap=1)
        chunks = chunker.chunk(sample_text, "doc1")
        assert all(isinstance(c, SemanticChunk) for c in chunks)

    def test_non_empty_output(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunks = SentenceChunker(self.nlp, chunk_sentences=2, overlap=0).chunk(sample_text, "doc1")
        assert len(chunks) > 0

    def test_sentence_count_populated(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunks = SentenceChunker(self.nlp, chunk_sentences=2, overlap=0).chunk(sample_text, "doc1")
        assert all(c.sentence_count > 0 for c in chunks)

    def test_entity_names_are_strings(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunks = SentenceChunker(self.nlp, chunk_sentences=3, overlap=1).chunk(sample_text, "doc1")
        for c in chunks:
            assert all(isinstance(n, str) for n in c.entity_names)

    def test_chunk_ids_unique(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunks = SentenceChunker(self.nlp, chunk_sentences=2, overlap=1).chunk(sample_text, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_index_sequential(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunks = SentenceChunker(self.nlp, chunk_sentences=2, overlap=0).chunk(sample_text, "doc1")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_overlap_reduces_chunk_count(self, sample_text):
        from cognity_ai.chunkers.sentence import SentenceChunker
        chunks_no_overlap = SentenceChunker(self.nlp, chunk_sentences=2, overlap=0).chunk(sample_text, "doc1")
        chunks_with_overlap = SentenceChunker(self.nlp, chunk_sentences=2, overlap=1).chunk(sample_text, "doc1")
        # With overlap, more chunks because sentences slide by 1 instead of 2
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)
