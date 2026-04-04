"""Trie (prefix tree) data structures.

Provides:
- TrieNode  — internal node used by Trie
- Trie      — full prefix tree with traversal, autocomplete, and LPM
- EntityTrie — subclass that maps lowercased keys → original-case values
"""
from __future__ import annotations

from collections import deque
from typing import Iterator


class TrieNode:
    """A single node in the trie."""

    __slots__ = ("children", "is_end", "original")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        # Stores original-case value when this node ends a word (used by EntityTrie)
        self.original: str | None = None


class Trie:
    """Prefix tree supporting O(k) insert, search, delete, and autocomplete.

    All operations are O(k) where k = len(key).
    """

    def __init__(self) -> None:
        self._root = TrieNode()
        self._count = 0

    # ── Core operations ───────────────────────────────────────────────────────

    def insert(self, word: str) -> None:
        """Insert *word* into the trie."""
        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_end:
            node.is_end = True
            self._count += 1

    def search(self, word: str) -> bool:
        """Return True if *word* is in the trie (exact match)."""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word in the trie starts with *prefix*."""
        return self._find_node(prefix) is not None

    def delete(self, word: str) -> bool:
        """Remove *word* from the trie.  Returns True if the word existed."""
        return self._delete(self._root, word, 0)

    # ── Enumeration ───────────────────────────────────────────────────────────

    def words_with_prefix(self, prefix: str, max_results: int = 100) -> list[str]:
        """Return up to *max_results* words that start with *prefix* (DFS)."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results: list[str] = []
        self._dfs_collect(node, prefix, results, max_results)
        return results

    def autocomplete(self, prefix: str, max_results: int = 10) -> list[str]:
        """Alias for :meth:`words_with_prefix`."""
        return self.words_with_prefix(prefix, max_results)

    def all_words(self) -> list[str]:
        """Return all words in BFS level order."""
        words: list[str] = []
        queue: deque[tuple[TrieNode, str]] = deque([(self._root, "")])
        while queue:
            node, prefix = queue.popleft()
            if node.is_end:
                words.append(prefix)
            for ch, child in node.children.items():
                queue.append((child, prefix + ch))
        return words

    def count_words(self) -> int:
        """Return the number of distinct words stored."""
        return self._count

    # ── Longest prefix match ──────────────────────────────────────────────────

    def longest_prefix_match(self, query: str) -> str:
        """Return the longest stored prefix that is a prefix of *query*.

        Returns an empty string when no prefix matches.
        """
        node = self._root
        last_match = ""
        current = ""
        for ch in query:
            if ch not in node.children:
                break
            node = node.children[ch]
            current += ch
            if node.is_end:
                last_match = current
        return last_match

    # ── Graph traversals (for debugging / introspection) ─────────────────────

    def bfs_nodes(self) -> list[tuple[str, bool]]:
        """BFS over all trie nodes.

        Returns a list of ``(prefix, is_end)`` tuples in breadth-first order.
        """
        result: list[tuple[str, bool]] = []
        queue: deque[tuple[TrieNode, str]] = deque([(self._root, "")])
        while queue:
            node, prefix = queue.popleft()
            result.append((prefix, node.is_end))
            for ch, child in node.children.items():
                queue.append((child, prefix + ch))
        return result

    def dfs_nodes(self) -> list[tuple[str, bool]]:
        """DFS (pre-order) over all trie nodes.

        Returns a list of ``(word_so_far, is_end)`` tuples.
        """
        result: list[tuple[str, bool]] = []
        self._dfs_nodes(self._root, "", result)
        return result

    def __len__(self) -> int:
        return self._count

    def __contains__(self, word: str) -> bool:
        return self.search(word)

    def __iter__(self) -> Iterator[str]:
        return iter(self.all_words())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_node(self, prefix: str) -> TrieNode | None:
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _dfs_collect(
        self,
        node: TrieNode,
        current: str,
        results: list[str],
        max_results: int,
    ) -> None:
        if len(results) >= max_results:
            return
        if node.is_end:
            results.append(current)
        for ch, child in node.children.items():
            if len(results) >= max_results:
                return
            self._dfs_collect(child, current + ch, results, max_results)

    def _dfs_nodes(
        self,
        node: TrieNode,
        current: str,
        result: list[tuple[str, bool]],
    ) -> None:
        result.append((current, node.is_end))
        for ch, child in node.children.items():
            self._dfs_nodes(child, current + ch, result)

    def _delete(self, node: TrieNode, word: str, depth: int) -> bool:
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            node.original = None
            self._count -= 1
            return True
        ch = word[depth]
        if ch not in node.children:
            return False
        deleted = self._delete(node.children[ch], word, depth + 1)
        if deleted:
            child = node.children[ch]
            if not child.children and not child.is_end:
                del node.children[ch]
        return deleted


class EntityTrie(Trie):
    """Trie specialised for entity names.

    Keys are stored lowercased for case-insensitive lookup, but the original-
    case name is preserved in the terminal node so it can be returned to
    callers.
    """

    def insert_entity(self, name: str) -> None:
        """Insert *name* using its lowercased form as the key."""
        key = name.lower()
        node = self._root
        for ch in key:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        if not node.is_end:
            node.is_end = True
            self._count += 1
        # Always update original to reflect the latest casing
        node.original = name

    def delete_entity(self, name: str) -> bool:
        """Remove *name* (case-insensitive)."""
        return self.delete(name.lower())

    def search_entities(self, prefix: str, max_results: int = 100) -> list[str]:
        """Return original-case entity names matching *prefix* (case-insensitive)."""
        key = prefix.lower()
        node = self._find_node(key)
        if node is None:
            return []
        results: list[str] = []
        self._collect_originals(node, results, max_results)
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _collect_originals(
        self,
        node: TrieNode,
        results: list[str],
        max_results: int,
    ) -> None:
        if len(results) >= max_results:
            return
        if node.is_end:
            results.append(node.original or "")
        for child in node.children.values():
            if len(results) >= max_results:
                return
            self._collect_originals(child, results, max_results)
