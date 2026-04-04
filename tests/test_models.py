"""Tests for Pydantic data models: Document, SemanticChunk, knowledge models, etc."""
from __future__ import annotations

import pytest
from cognity_ai.models.document import Document, ImageRef
from cognity_ai.models.retrieval import (
    PageInfo, SemanticChunk, CommunityInfo, DocumentMeta, RetrievalResult,
)
from cognity_ai.models.knowledge import Entity, Relation, ExtractionResult, SourceStatus


# ── Document ─────────────────────────────────────────────────────────────────

class TestDocument:
    def test_minimal_creation(self):
        doc = Document(doc_id="doc1", text="Hello world")
        assert doc.doc_id == "doc1"
        assert doc.text == "Hello world"

    def test_char_count_auto_computed(self):
        doc = Document(doc_id="d", text="abc")
        assert doc.char_count == 3

    def test_char_count_not_overwritten_when_set(self):
        doc = Document(doc_id="d", text="abc", char_count=99)
        assert doc.char_count == 99

    def test_file_extension_from_source_path(self):
        doc = Document(doc_id="d", text="hi", source_path="/tmp/report.pdf")
        assert doc.file_extension == ".pdf"

    def test_file_extension_not_overwritten_when_set(self):
        doc = Document(doc_id="d", text="hi", source_path="/tmp/x.pdf", file_extension=".txt")
        assert doc.file_extension == ".txt"

    def test_defaults(self):
        doc = Document(doc_id="d", text="")
        assert doc.source_path == ""
        assert doc.source_name == ""
        assert doc.loader == ""
        assert doc.metadata == {}
        assert doc.page_map == []
        assert doc.image_refs == []
        assert doc.file_size_bytes == 0
        assert doc.page_count == 0

    def test_metadata_dict(self):
        doc = Document(doc_id="d", text="t", metadata={"author": "Alice"})
        assert doc.metadata["author"] == "Alice"

    def test_page_map(self):
        pm = [{"page_num": 1, "start_char": 0, "end_char": 100, "heading": "Intro"}]
        doc = Document(doc_id="d", text="t" * 100, page_map=pm)
        assert len(doc.page_map) == 1
        assert doc.page_map[0]["heading"] == "Intro"


class TestImageRef:
    def test_minimal(self):
        ref = ImageRef(image_id="img1", char_offset=50)
        assert ref.image_id == "img1"
        assert ref.char_offset == 50
        assert ref.mime_type == "image/png"
        assert ref.ocr_text == ""

    def test_with_bytes(self):
        ref = ImageRef(image_id="img2", char_offset=0, image_bytes=b"\x89PNG")
        assert ref.image_bytes == b"\x89PNG"


# ── PageInfo ─────────────────────────────────────────────────────────────────

class TestPageInfo:
    def test_creation(self):
        pi = PageInfo(page_num=3, section="Methods", start_char=100, end_char=500, heading="Methods")
        assert pi.page_num == 3
        assert pi.section == "Methods"
        assert pi.start_char == 100
        assert pi.end_char == 500

    def test_defaults(self):
        pi = PageInfo(page_num=1)
        assert pi.section == ""
        assert pi.start_char == 0
        assert pi.end_char == 0
        assert pi.heading == ""


# ── SemanticChunk ─────────────────────────────────────────────────────────────

class TestSemanticChunk:
    def test_minimal(self):
        chunk = SemanticChunk(chunk_id="doc1__chunk_0", doc_id="doc1", text="hello", index=0)
        assert chunk.chunk_id == "doc1__chunk_0"
        assert chunk.doc_id == "doc1"
        assert chunk.text == "hello"
        assert chunk.index == 0

    def test_defaults(self):
        chunk = SemanticChunk(chunk_id="c", doc_id="d", text="t", index=0)
        assert chunk.embedding is None
        assert chunk.entity_names == []
        assert chunk.sentence_count == 0
        assert chunk.token_estimate == 0
        assert chunk.parent_chunk_id is None
        assert chunk.is_parent is False
        assert chunk.page_info is None

    def test_with_embedding(self):
        chunk = SemanticChunk(chunk_id="c", doc_id="d", text="t", index=0, embedding=[0.1, 0.2, 0.3])
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_with_page_info(self):
        pi = PageInfo(page_num=2, start_char=50, end_char=200)
        chunk = SemanticChunk(chunk_id="c", doc_id="d", text="t", index=0, page_info=pi)
        assert chunk.page_info.page_num == 2

    def test_entity_names(self):
        chunk = SemanticChunk(chunk_id="c", doc_id="d", text="t", index=0,
                              entity_names=["Alice", "Bob"])
        assert "Alice" in chunk.entity_names

    def test_parent_child_flags(self):
        parent = SemanticChunk(chunk_id="c_parent", doc_id="d", text="t", index=0, is_parent=True)
        child = SemanticChunk(chunk_id="c_child", doc_id="d", text="t", index=1,
                              parent_chunk_id="c_parent")
        assert parent.is_parent is True
        assert child.parent_chunk_id == "c_parent"


# ── CommunityInfo ─────────────────────────────────────────────────────────────

class TestCommunityInfo:
    def test_creation(self):
        comm = CommunityInfo(community_id="c1", level=0,
                             entity_names=["Alice", "Bob"],
                             summary="Key collaboration community",
                             title="Collab Group")
        assert comm.community_id == "c1"
        assert comm.level == 0
        assert "Alice" in comm.entity_names
        assert comm.title == "Collab Group"

    def test_defaults(self):
        comm = CommunityInfo(community_id="c1", level=1)
        assert comm.entity_names == []
        assert comm.summary == ""
        assert comm.title == ""
        assert comm.parent_community is None
        assert comm.rank == 0.0
        assert comm.embedding is None


# ── RetrievalResult ───────────────────────────────────────────────────────────

class TestRetrievalResult:
    def test_creation(self):
        rr = RetrievalResult(content="Some text", score=0.92, source="vector")
        assert rr.content == "Some text"
        assert rr.score == 0.92
        assert rr.source == "vector"
        assert rr.metadata == {}

    def test_with_metadata(self):
        rr = RetrievalResult(content="t", score=0.5, source="graph",
                             metadata={"doc_id": "doc1", "page_num": 3})
        assert rr.metadata["doc_id"] == "doc1"

    @pytest.mark.parametrize("source", ["vector", "graph", "community", "page", "vector_bridge"])
    def test_valid_sources(self, source):
        rr = RetrievalResult(content="x", score=0.5, source=source)
        assert rr.source == source


# ── Knowledge models ──────────────────────────────────────────────────────────

class TestEntity:
    def test_creation(self):
        e = Entity(name="Alice", entity_type="Person", description="A researcher")
        assert e.name == "Alice"
        assert e.entity_type == "Person"

    def test_defaults(self):
        e = Entity(name="X", entity_type="Other")
        assert e.description == ""
        assert e.properties == {}
        assert e.source_id == ""
        assert e.confidence == 1.0
        assert e.extraction_method == "nlp"
        assert e.mentions == 1


class TestRelation:
    def test_creation(self):
        r = Relation(source_entity="Alice", relation_type="WORKS_AT", target_entity="Acme")
        assert r.source_entity == "Alice"
        assert r.relation_type == "WORKS_AT"
        assert r.target_entity == "Acme"

    def test_defaults(self):
        r = Relation(source_entity="A", relation_type="REL", target_entity="B")
        assert r.description == ""
        assert r.confidence == 1.0
        assert r.weight == 1.0
        assert r.extraction_method == "nlp"


class TestExtractionResult:
    def test_empty(self):
        er = ExtractionResult()
        assert er.entities == []
        assert er.relations == []

    def test_with_data(self):
        e = Entity(name="Alice", entity_type="Person")
        r = Relation(source_entity="Alice", relation_type="KNOWS", target_entity="Bob")
        er = ExtractionResult(entities=[e], relations=[r])
        assert len(er.entities) == 1
        assert len(er.relations) == 1


class TestSourceStatus:
    def test_values(self):
        assert SourceStatus.PENDING == "pending"
        assert SourceStatus.CONFIRMED == "confirmed"
        assert SourceStatus.DEPRECATED == "deprecated"
