"""Tests for graph store backends.

NetworkX (in-memory, zero extra deps) is tested directly.
Neo4j / Memgraph / ArangoDB are skipped unless credentials are available.
"""
from __future__ import annotations

import os
import pytest
from conftest import requires_networkx
from cognity_ai.models.knowledge import Entity, Relation
from cognity_ai.models.retrieval import CommunityInfo, RetrievalResult


# ── NetworkXStore ─────────────────────────────────────────────────────────────

@requires_networkx
class TestNetworkXStore:
    @pytest.fixture
    def store(self):
        from cognity_ai.stores.graph.networkx import NetworkXStore
        return NetworkXStore()

    # Entity operations
    def test_upsert_entity(self, store):
        e = Entity(name="Alice", entity_type="Person", description="A researcher")
        store.upsert_entity(e)
        report = store.health_report()
        assert report["entities"] == 1

    def test_upsert_entity_merges_on_revisit(self, store):
        e1 = Entity(name="Alice", entity_type="Person", mentions=1)
        e2 = Entity(name="Alice", entity_type="Person", mentions=2)
        store.upsert_entity(e1)
        store.upsert_entity(e2)
        assert store.health_report()["entities"] == 1

    def test_upsert_multiple_entities(self, store):
        for name in ["Alice", "Bob", "Carol"]:
            store.upsert_entity(Entity(name=name, entity_type="Person"))
        assert store.health_report()["entities"] == 3

    # Relation operations
    def test_upsert_relation(self, store):
        store.upsert_entity(Entity(name="Alice", entity_type="Person"))
        store.upsert_entity(Entity(name="Acme", entity_type="Organization"))
        store.upsert_relation(Relation(source_entity="Alice", relation_type="WORKS_AT",
                                       target_entity="Acme"))
        assert store.health_report()["relations"] >= 1

    def test_upsert_multiple_relations(self, store):
        for src, rel, tgt in [("Alice", "KNOWS", "Bob"), ("Bob", "WORKS_AT", "Acme")]:
            store.upsert_entity(Entity(name=src, entity_type="Person"))
            store.upsert_entity(Entity(name=tgt, entity_type="Organization"))
            store.upsert_relation(Relation(source_entity=src, relation_type=rel,
                                            target_entity=tgt))
        assert store.health_report()["relations"] >= 2

    # Chunk-entity linking
    def test_link_chunk_to_entities(self, store):
        store.upsert_entity(Entity(name="Alice", entity_type="Person"))
        store.link_chunk_to_entities("chunk_1", "doc1", ["Alice"])
        chunks = store.get_chunks_for_entities(["Alice"])
        assert "chunk_1" in chunks

    def test_get_chunks_for_missing_entity(self, store):
        result = store.get_chunks_for_entities(["Unknown"])
        assert result == []

    def test_multiple_entities_linked_to_chunk(self, store):
        for name in ["Alice", "Bob"]:
            store.upsert_entity(Entity(name=name, entity_type="Person"))
        store.link_chunk_to_entities("chunk_1", "doc1", ["Alice", "Bob"])
        alice_chunks = store.get_chunks_for_entities(["Alice"])
        bob_chunks = store.get_chunks_for_entities(["Bob"])
        assert "chunk_1" in alice_chunks
        assert "chunk_1" in bob_chunks

    # Retrieval
    def test_retrieve_subgraph(self, store):
        store.upsert_entity(Entity(name="Alice", entity_type="Person"))
        store.upsert_entity(Entity(name="Acme", entity_type="Organization"))
        store.upsert_relation(Relation(source_entity="Alice", relation_type="WORKS_AT",
                                       target_entity="Acme"))
        results = store.retrieve_subgraph(["Alice"], hops=1)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_subgraph_unknown_entity_returns_empty(self, store):
        results = store.retrieve_subgraph(["Unknown"], hops=2)
        assert results == []

    def test_retrieve_entity_context(self, store):
        store.upsert_entity(Entity(name="Alice", entity_type="Person",
                                   description="A researcher"))
        store.upsert_entity(Entity(name="Acme", entity_type="Organization"))
        store.upsert_relation(Relation(source_entity="Alice", relation_type="WORKS_AT",
                                       target_entity="Acme"))
        results = store.retrieve_entity_context("Alice")
        assert len(results) >= 1
        assert "Alice" in results[0].content

    def test_retrieve_entity_context_unknown(self, store):
        results = store.retrieve_entity_context("Nobody")
        assert results == []

    # Doc metadata
    def test_upsert_doc_meta(self, store):
        store.upsert_doc_meta("doc1", "hash123", "report.pdf", "pending")
        assert store.get_doc_status("doc1") == "pending"

    def test_confirm_source(self, store):
        store.upsert_doc_meta("doc1", "h", "f.pdf", "pending")
        store.confirm_source("doc1")
        assert store.get_doc_status("doc1") == "confirmed"

    def test_deprecate_source(self, store):
        store.upsert_doc_meta("doc1", "h", "f.pdf", "confirmed")
        store.upsert_entity(Entity(name="Alice", entity_type="Person"))
        store.upsert_entity(Entity(name="Acme", entity_type="Organization"))
        store.upsert_relation(Relation(source_entity="Alice", relation_type="WORKS_AT",
                                       target_entity="Acme", source_id="doc1",
                                       confidence=1.0))
        store.deprecate_source("doc1")
        assert store.get_doc_status("doc1") == "deprecated"

    def test_get_doc_status_missing(self, store):
        assert store.get_doc_status("nope") is None

    # Community operations
    def test_store_community_summary(self, store):
        comm = CommunityInfo(community_id="c1", level=0,
                             title="AI Research", summary="Covers AI",
                             entity_names=["Alice", "Bob"], rank=0.8)
        store.store_community_summary(comm)
        assert store.health_report()["communities"] == 1

    def test_global_community_search(self, store):
        for i in range(3):
            comm = CommunityInfo(community_id=f"c{i}", level=0,
                                 title=f"Community {i}", summary=f"Summary {i}",
                                 rank=float(i) / 3)
            store.store_community_summary(comm)
        results = store.global_community_search(top_n=2)
        assert len(results) <= 2
        assert all(r.source == "community" for r in results)

    # Graph analysis
    def test_detect_communities_empty_graph(self, store):
        result = store.detect_communities()
        assert result == []

    def test_detect_communities_with_nodes(self, store):
        for name in ["Alice", "Bob", "Carol", "Dave"]:
            store.upsert_entity(Entity(name=name, entity_type="Person"))
        for src, tgt in [("Alice", "Bob"), ("Bob", "Carol"), ("Carol", "Dave")]:
            store.upsert_relation(Relation(source_entity=src, relation_type="KNOWS",
                                           target_entity=tgt))
        result = store.detect_communities()
        assert isinstance(result, list)

    # Confidence pruning
    def test_prune_low_confidence(self, store):
        store.upsert_entity(Entity(name="A", entity_type="X"))
        store.upsert_entity(Entity(name="B", entity_type="X"))
        store.upsert_relation(Relation(source_entity="A", relation_type="REL",
                                       target_entity="B", confidence=0.1))
        removed = store.prune_low_confidence(threshold=0.5)
        assert removed == 1

    def test_prune_keeps_high_confidence(self, store):
        store.upsert_entity(Entity(name="A", entity_type="X"))
        store.upsert_entity(Entity(name="B", entity_type="X"))
        store.upsert_relation(Relation(source_entity="A", relation_type="REL",
                                       target_entity="B", confidence=0.9))
        removed = store.prune_low_confidence(threshold=0.5)
        assert removed == 0

    # Health report
    def test_health_report_structure(self, store):
        report = store.health_report()
        expected_keys = {"entities", "relations", "documents", "confirmed_sources",
                         "communities", "avg_confidence"}
        assert expected_keys.issubset(set(report.keys()))


# ── Neo4j (skip unless NEO4J_URI + credentials available) ────────────────────

class TestNeo4jStore:
    @pytest.fixture(autouse=True)
    def _require_neo4j(self):
        pytest.importorskip("neo4j")
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_pass = os.getenv("NEO4J_PASSWORD")
        if not (neo4j_uri and neo4j_pass):
            pytest.skip("NEO4J_URI and NEO4J_PASSWORD not set")

    @pytest.fixture
    def store(self):
        from cognity_ai.stores.graph.neo4j import Neo4jStore
        from cognity_ai.config.providers import Neo4jConfig, GraphRAGConfig
        cfg = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
        )
        try:
            return Neo4jStore(cfg, GraphRAGConfig())
        except Exception:
            pytest.skip("Could not connect to Neo4j")

    def test_upsert_entity_and_retrieve(self, store):
        e = Entity(name="TestAlice", entity_type="Person")
        store.upsert_entity(e)
        results = store.retrieve_entity_context("TestAlice")
        assert isinstance(results, list)

    def test_upsert_relation(self, store):
        store.upsert_entity(Entity(name="TestAlice2", entity_type="Person"))
        store.upsert_entity(Entity(name="TestCorp", entity_type="Organization"))
        store.upsert_relation(Relation(source_entity="TestAlice2",
                                        relation_type="TEST_WORKS_AT",
                                        target_entity="TestCorp"))
        results = store.retrieve_subgraph(["TestAlice2"], hops=1)
        assert isinstance(results, list)
