"""Tests for cognity_ai.utils.trie (Trie and EntityTrie)."""
from __future__ import annotations

import pytest
from cognity_ai.utils.trie import Trie, EntityTrie


# ── Trie — basic operations ───────────────────────────────────────────────────

class TestTrieInsertSearch:
    def test_search_missing_word(self):
        t = Trie()
        assert t.search("apple") is False

    def test_insert_then_search(self):
        t = Trie()
        t.insert("apple")
        assert t.search("apple") is True

    def test_search_is_exact_match(self):
        t = Trie()
        t.insert("apple")
        assert t.search("app") is False

    def test_starts_with_true(self):
        t = Trie()
        t.insert("application")
        assert t.starts_with("app") is True

    def test_starts_with_false(self):
        t = Trie()
        t.insert("banana")
        assert t.starts_with("app") is False

    def test_insert_multiple_words(self):
        t = Trie()
        for w in ["apple", "app", "apply", "apt"]:
            t.insert(w)
        for w in ["apple", "app", "apply", "apt"]:
            assert t.search(w) is True

    def test_count_words(self):
        t = Trie()
        for w in ["a", "b", "c"]:
            t.insert(w)
        assert t.count_words() == 3

    def test_count_words_no_duplicates(self):
        t = Trie()
        t.insert("hello")
        t.insert("hello")
        assert t.count_words() == 1

    def test_len(self):
        t = Trie()
        t.insert("x")
        t.insert("y")
        assert len(t) == 2

    def test_contains_operator(self):
        t = Trie()
        t.insert("foo")
        assert "foo" in t
        assert "bar" not in t


# ── Trie — enumeration ────────────────────────────────────────────────────────

class TestTrieEnumeration:
    WORDS = ["apple", "application", "apply", "apt", "banana"]

    @pytest.fixture
    def t(self):
        trie = Trie()
        for w in self.WORDS:
            trie.insert(w)
        return trie

    def test_words_with_prefix_app(self, t):
        result = t.words_with_prefix("app")
        assert set(result) == {"apple", "application", "apply"}

    def test_words_with_prefix_empty(self, t):
        result = t.words_with_prefix("")
        assert set(result) == set(self.WORDS)

    def test_words_with_prefix_no_match(self, t):
        assert t.words_with_prefix("xyz") == []

    def test_words_with_prefix_max_results(self, t):
        result = t.words_with_prefix("app", max_results=2)
        assert len(result) == 2

    def test_autocomplete_alias(self, t):
        assert t.autocomplete("app") == t.words_with_prefix("app")

    def test_all_words_contains_all(self, t):
        assert set(t.all_words()) == set(self.WORDS)

    def test_iter(self, t):
        assert set(iter(t)) == set(self.WORDS)


# ── Trie — delete ─────────────────────────────────────────────────────────────

class TestTrieDelete:
    def test_delete_existing_word(self):
        t = Trie()
        t.insert("hello")
        assert t.delete("hello") is True
        assert t.search("hello") is False

    def test_delete_missing_word_returns_false(self):
        t = Trie()
        assert t.delete("ghost") is False

    def test_delete_decrements_count(self):
        t = Trie()
        t.insert("hello")
        t.delete("hello")
        assert t.count_words() == 0

    def test_delete_prefix_of_another_word(self):
        t = Trie()
        t.insert("apple")
        t.insert("app")
        t.delete("app")
        assert t.search("app") is False
        assert t.search("apple") is True

    def test_delete_pruning(self):
        t = Trie()
        t.insert("abcde")
        t.delete("abcde")
        # After deleting the only word, root should have no children
        assert t._root.children == {}


# ── Trie — longest prefix match ───────────────────────────────────────────────

class TestLongestPrefixMatch:
    def test_exact_match(self):
        t = Trie()
        t.insert("apple")
        assert t.longest_prefix_match("apple") == "apple"

    def test_partial_match(self):
        t = Trie()
        t.insert("app")
        assert t.longest_prefix_match("application") == "app"

    def test_no_match(self):
        t = Trie()
        t.insert("banana")
        assert t.longest_prefix_match("apple") == ""

    def test_longest_wins(self):
        t = Trie()
        t.insert("app")
        t.insert("appl")
        assert t.longest_prefix_match("application") == "appl"

    def test_empty_query(self):
        t = Trie()
        t.insert("abc")
        assert t.longest_prefix_match("") == ""


# ── Trie — traversal ──────────────────────────────────────────────────────────

class TestTrieTraversals:
    @pytest.fixture
    def t(self):
        trie = Trie()
        for w in ["a", "ab", "abc"]:
            trie.insert(w)
        return trie

    def test_bfs_nodes_includes_root(self, t):
        nodes = t.bfs_nodes()
        assert nodes[0] == ("", False)

    def test_bfs_nodes_all_end_nodes(self, t):
        end_nodes = {prefix for prefix, is_end in t.bfs_nodes() if is_end}
        assert end_nodes == {"a", "ab", "abc"}

    def test_dfs_nodes_has_all_words(self, t):
        end_nodes = {prefix for prefix, is_end in t.dfs_nodes() if is_end}
        assert end_nodes == {"a", "ab", "abc"}

    def test_bfs_and_dfs_cover_same_words(self, t):
        bfs_words = {p for p, e in t.bfs_nodes() if e}
        dfs_words = {p for p, e in t.dfs_nodes() if e}
        assert bfs_words == dfs_words


# ── EntityTrie ────────────────────────────────────────────────────────────────

class TestEntityTrie:
    def test_insert_and_search_exact(self):
        et = EntityTrie()
        et.insert_entity("Alice Smith")
        assert et.search_entities("alice") == ["Alice Smith"]

    def test_search_is_case_insensitive(self):
        et = EntityTrie()
        et.insert_entity("Bob Jones")
        assert et.search_entities("BOB") == ["Bob Jones"]
        assert et.search_entities("bob") == ["Bob Jones"]

    def test_returns_original_case(self):
        et = EntityTrie()
        et.insert_entity("OpenAI")
        result = et.search_entities("open")
        assert result == ["OpenAI"]

    def test_multiple_matches(self):
        et = EntityTrie()
        et.insert_entity("Alice")
        et.insert_entity("Alan Turing")
        result = et.search_entities("al")
        assert set(result) == {"Alice", "Alan Turing"}

    def test_no_match(self):
        et = EntityTrie()
        et.insert_entity("Charlie")
        assert et.search_entities("xyz") == []

    def test_delete_entity(self):
        et = EntityTrie()
        et.insert_entity("Dave")
        et.delete_entity("Dave")
        assert et.search_entities("dave") == []

    def test_max_results_respected(self):
        et = EntityTrie()
        for i in range(20):
            et.insert_entity(f"Entity{i:02d}")
        result = et.search_entities("entity", max_results=5)
        assert len(result) == 5

    def test_insert_updates_original_casing(self):
        et = EntityTrie()
        et.insert_entity("openai")
        et.insert_entity("OpenAI")  # reinsertion with different case
        result = et.search_entities("openai")
        assert result == ["OpenAI"]

    def test_count_words(self):
        et = EntityTrie()
        et.insert_entity("A")
        et.insert_entity("B")
        assert et.count_words() == 2

    def test_empty_prefix_returns_all(self):
        et = EntityTrie()
        names = ["Alice", "Bob", "Charlie"]
        for n in names:
            et.insert_entity(n)
        result = et.search_entities("", max_results=100)
        assert set(result) == set(names)
