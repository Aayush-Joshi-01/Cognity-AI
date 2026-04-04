"""Tests for document loaders (TxtLoader, MdLoader, and optional format loaders)."""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from cognity_ai.loaders.text import TxtLoader, MdLoader
from cognity_ai.models.document import Document


# ── TxtLoader ─────────────────────────────────────────────────────────────────

class TestTxtLoader:
    def test_loads_txt_file(self, sample_text_file, sample_text):
        loader = TxtLoader()
        docs = loader.load(sample_text_file)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].text == sample_text

    def test_loader_name(self, sample_text_file):
        docs = TxtLoader().load(sample_text_file)
        assert docs[0].loader == "TxtLoader"

    def test_source_name_is_filename(self, sample_text_file):
        docs = TxtLoader().load(sample_text_file)
        assert docs[0].source_name == "sample.txt"

    def test_file_extension(self, sample_text_file):
        docs = TxtLoader().load(sample_text_file)
        assert docs[0].file_extension == ".txt"

    def test_file_size_nonzero(self, sample_text_file):
        docs = TxtLoader().load(sample_text_file)
        assert docs[0].file_size_bytes > 0

    def test_page_count_is_one(self, sample_text_file):
        docs = TxtLoader().load(sample_text_file)
        assert docs[0].page_count == 1

    def test_page_map_has_one_entry(self, sample_text_file):
        docs = TxtLoader().load(sample_text_file)
        assert len(docs[0].page_map) == 1
        pm = docs[0].page_map[0]
        assert pm["page_num"] == 1
        assert pm["start_char"] == 0

    def test_doc_id_is_unique(self, tmp_path, sample_text):
        p = tmp_path / "sample.txt"
        p.write_text(sample_text, encoding="utf-8")
        loader = TxtLoader()
        docs1 = loader.load(str(p))
        docs2 = loader.load(str(p))
        assert docs1[0].doc_id != docs2[0].doc_id

    def test_char_count_matches_text(self, sample_text_file, sample_text):
        docs = TxtLoader().load(sample_text_file)
        assert docs[0].char_count == len(sample_text)

    def test_supported_extensions(self):
        assert ".txt" in TxtLoader().supported_extensions

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        docs = TxtLoader().load(str(p))
        assert len(docs) == 1
        assert docs[0].text == ""
        assert docs[0].char_count == 0


# ── MdLoader ─────────────────────────────────────────────────────────────────

class TestMdLoader:
    def test_loads_md_file(self, sample_md_file):
        loader = MdLoader()
        docs = loader.load(sample_md_file)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Introduction" in docs[0].text

    def test_loader_name(self, sample_md_file):
        docs = MdLoader().load(sample_md_file)
        assert docs[0].loader == "MdLoader"

    def test_file_extension(self, sample_md_file):
        docs = MdLoader().load(sample_md_file)
        assert docs[0].file_extension == ".md"

    def test_page_map_built_from_headings(self, sample_md_file):
        docs = MdLoader().load(sample_md_file)
        pm = docs[0].page_map
        headings = [entry["heading"] for entry in pm]
        assert "Introduction" in headings
        assert "Methods" in headings
        assert "Results" in headings

    def test_page_count_matches_headings(self, sample_md_file):
        docs = MdLoader().load(sample_md_file)
        assert docs[0].page_count >= 3

    def test_page_map_start_end_chars_ordered(self, sample_md_file):
        docs = MdLoader().load(sample_md_file)
        pm = docs[0].page_map
        for entry in pm:
            assert entry["start_char"] < entry["end_char"]

    def test_no_headings_gives_single_section(self, tmp_path):
        p = tmp_path / "flat.md"
        p.write_text("Just some text, no headings here.", encoding="utf-8")
        docs = MdLoader().load(str(p))
        assert len(docs[0].page_map) == 1

    def test_preamble_before_first_heading_is_captured(self, tmp_path):
        content = "Preamble text here.\n\n# First Heading\n\nBody.\n"
        p = tmp_path / "preamble.md"
        p.write_text(content, encoding="utf-8")
        docs = MdLoader().load(str(p))
        pm = docs[0].page_map
        # First entry should be the preamble (page_num 0) or the heading
        page_nums = [e["page_num"] for e in pm]
        assert 0 in page_nums or "First Heading" in [e["heading"] for e in pm]

    def test_supported_extensions(self):
        exts = MdLoader().supported_extensions
        assert ".md" in exts
        assert ".markdown" in exts

    def test_source_path_is_absolute(self, sample_md_file):
        docs = MdLoader().load(sample_md_file)
        assert os.path.isabs(docs[0].source_path)


# ── LoaderFactory ─────────────────────────────────────────────────────────────

class TestLoaderFactory:
    def test_txt_dispatched_correctly(self, sample_text_file):
        from cognity_ai.loaders.factory import LoaderFactory
        loader = LoaderFactory.get_loader(sample_text_file)
        assert isinstance(loader, TxtLoader)

    def test_md_dispatched_correctly(self, sample_md_file):
        from cognity_ai.loaders.factory import LoaderFactory
        loader = LoaderFactory.get_loader(sample_md_file)
        assert isinstance(loader, MdLoader)

    def test_unknown_extension_raises(self, tmp_path):
        from cognity_ai.loaders.factory import LoaderFactory
        p = tmp_path / "data.xyz"
        p.write_text("data")
        with pytest.raises(Exception):
            LoaderFactory.get_loader(str(p))

    def test_load_convenience_method(self, sample_text_file, sample_text):
        from cognity_ai.loaders.factory import LoaderFactory
        docs = LoaderFactory.load(sample_text_file)
        assert len(docs) == 1
        assert docs[0].text == sample_text

    def test_load_with_custom_doc_id(self, sample_text_file):
        from cognity_ai.loaders.factory import LoaderFactory
        docs = LoaderFactory.load(sample_text_file, doc_id="my-doc-id")
        assert docs[0].doc_id == "my-doc-id"

    def test_load_with_extra_metadata(self, sample_text_file):
        from cognity_ai.loaders.factory import LoaderFactory
        docs = LoaderFactory.load(sample_text_file, author="Alice")
        assert docs[0].metadata.get("author") == "Alice"

    def test_supported_extensions_includes_common(self):
        from cognity_ai.loaders.factory import LoaderFactory
        exts = LoaderFactory.supported_extensions()
        assert ".txt" in exts
        assert ".pdf" in exts
        assert ".md" in exts


# ── Optional loaders (skip if deps missing) ───────────────────────────────────

class TestPdfLoader:
    @pytest.fixture(autouse=True)
    def _require_pdfplumber(self):
        pytest.importorskip("pdfplumber")

    def test_loader_importable(self):
        from cognity_ai.loaders.pdf import PdfLoader
        assert PdfLoader is not None

    def test_supported_extension(self):
        from cognity_ai.loaders.pdf import PdfLoader
        assert ".pdf" in PdfLoader().supported_extensions


class TestDocxLoader:
    @pytest.fixture(autouse=True)
    def _require_docx(self):
        pytest.importorskip("docx")

    def test_loader_importable(self):
        from cognity_ai.loaders.docx import DocxLoader
        assert DocxLoader is not None


class TestCsvLoader:
    @pytest.fixture(autouse=True)
    def _require_pandas(self):
        pytest.importorskip("pandas")

    def test_loader_importable(self):
        from cognity_ai.loaders.csv import CsvLoader
        assert CsvLoader is not None

    def test_loads_simple_csv(self, tmp_path):
        from cognity_ai.loaders.csv import CsvLoader
        p = tmp_path / "data.csv"
        p.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")
        docs = CsvLoader().load(str(p))
        assert len(docs) >= 1
        assert "Alice" in docs[0].text or "name" in docs[0].text


class TestHtmlLoader:
    @pytest.fixture(autouse=True)
    def _require_bs4(self):
        pytest.importorskip("bs4")

    def test_loader_importable(self):
        from cognity_ai.loaders.html import HtmlLoader
        assert HtmlLoader is not None

    def test_loads_simple_html(self, tmp_path):
        from cognity_ai.loaders.html import HtmlLoader
        p = tmp_path / "page.html"
        p.write_text("<html><body><h1>Hello</h1><p>World</p></body></html>", encoding="utf-8")
        docs = HtmlLoader().load(str(p))
        assert len(docs) >= 1
        assert "Hello" in docs[0].text or "World" in docs[0].text
