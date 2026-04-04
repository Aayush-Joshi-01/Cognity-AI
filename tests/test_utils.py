"""Tests for utility modules: hash store, RRF, token counter."""
from __future__ import annotations

import os
import pytest
from cognity_ai.utils.hash import content_hash, HashStore
from cognity_ai.utils.rrf import reciprocal_rank_fusion
from cognity_ai.models.retrieval import RetrievalResult


# ── content_hash ──────────────────────────────────────────────────────────────

class TestContentHash:
    def test_returns_string(self):
        h = content_hash("hello")
        assert isinstance(h, str)

    def test_sha256_length(self):
        assert len(content_hash("hello")) == 64

    def test_deterministic(self):
        assert content_hash("foo") == content_hash("foo")

    def test_different_texts_different_hashes(self):
        assert content_hash("foo") != content_hash("bar")

    def test_empty_string(self):
        h = content_hash("")
        assert len(h) == 64

    def test_unicode(self):
        h = content_hash("こんにちは")
        assert len(h) == 64


# ── HashStore ─────────────────────────────────────────────────────────────────

class TestHashStore:
    def test_get_missing_key(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        assert store.get("missing") is None

    def test_set_and_get(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        store.set("doc1", "abc123")
        assert store.get("doc1") == "abc123"

    def test_persistence(self, tmp_dir):
        path = os.path.join(tmp_dir, "hashes.json")
        store = HashStore(path=path)
        store.set("doc1", content_hash("some text"))
        # Reload from disk
        store2 = HashStore(path=path)
        assert store2.get("doc1") is not None

    def test_is_unchanged_true(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        text = "hello world"
        store.set("doc1", content_hash(text))
        assert store.is_unchanged("doc1", text) is True

    def test_is_unchanged_false_on_modification(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        store.set("doc1", content_hash("original"))
        assert store.is_unchanged("doc1", "modified") is False

    def test_is_unchanged_false_for_missing(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        assert store.is_unchanged("nonexistent", "any text") is False

    def test_remove(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        store.set("doc1", "hash1")
        store.remove("doc1")
        assert store.get("doc1") is None

    def test_remove_missing_key_no_error(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        store.remove("nope")  # should not raise

    def test_all_doc_ids(self, tmp_dir):
        store = HashStore(path=os.path.join(tmp_dir, "hashes.json"))
        store.set("doc1", "h1")
        store.set("doc2", "h2")
        ids = store.all_doc_ids()
        assert "doc1" in ids
        assert "doc2" in ids

    def test_file_created_on_save(self, tmp_dir):
        path = os.path.join(tmp_dir, "subdir", "hashes.json")
        store = HashStore(path=path)
        store.set("doc1", "hash1")
        assert os.path.exists(path)

    def test_corrupted_file_falls_back_to_empty(self, tmp_dir):
        path = os.path.join(tmp_dir, "hashes.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        store = HashStore(path=path)
        assert store.get("anything") is None


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────

def _rr(content: str, score: float, source: str = "vector") -> RetrievalResult:
    return RetrievalResult(content=content, score=score, source=source)


class TestRRF:
    def test_single_list_passthrough(self):
        results = [_rr("a", 0.9), _rr("b", 0.7), _rr("c", 0.5)]
        fused = reciprocal_rank_fusion(results)
        assert len(fused) == 3
        # Top item from rank 0 should still be first (highest RRF score)
        assert fused[0].content == "a"

    def test_two_lists_item_in_both_gets_boosted(self):
        list1 = [_rr("shared", 0.9), _rr("only_in_1", 0.8)]
        list2 = [_rr("shared", 0.8), _rr("only_in_2", 0.7)]
        fused = reciprocal_rank_fusion(list1, list2)
        # "shared" should rank first since it appears in both lists
        assert fused[0].content == "shared"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_single_empty_list(self):
        list1 = [_rr("a", 0.9)]
        fused = reciprocal_rank_fusion(list1, [])
        assert len(fused) == 1

    def test_weights_applied(self):
        list1 = [_rr("a", 0.9), _rr("b", 0.8)]
        list2 = [_rr("b", 0.9), _rr("a", 0.8)]
        # Give list2 double weight — "b" leads list2 so should win
        fused = reciprocal_rank_fusion(list1, list2, weights=[1.0, 2.0])
        assert fused[0].content == "b"

    def test_weights_wrong_length_raises(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([_rr("a", 0.9)], weights=[1.0, 2.0])

    def test_returns_descending_scores(self):
        results = [_rr(str(i), float(i)) for i in range(5)]
        fused = reciprocal_rank_fusion(results)
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_positive(self):
        results = [_rr("x", 0.5), _rr("y", 0.3)]
        fused = reciprocal_rank_fusion(results)
        assert all(r.score > 0 for r in fused)

    def test_custom_k(self):
        results = [_rr("a", 0.9), _rr("b", 0.5)]
        fused_low_k = reciprocal_rank_fusion(results, k=1)
        fused_high_k = reciprocal_rank_fusion(results, k=1000)
        # With low k, the rank gap between first and second is larger
        scores_low = [r.score for r in fused_low_k]
        scores_high = [r.score for r in fused_high_k]
        ratio_low = scores_low[0] / scores_low[1]
        ratio_high = scores_high[0] / scores_high[1]
        assert ratio_low > ratio_high
