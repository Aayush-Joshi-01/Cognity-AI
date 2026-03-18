"""
KnowledgeUpdater — lifecycle manager for knowledge sources.

Direct migration of hybrid_rag/knowledge_updater.py, refactored to
use the BaseGraphStore interface.
"""
from __future__ import annotations
from raglib.stores.graph.base import BaseGraphStore


class KnowledgeUpdater:
    def __init__(self, graph_store: BaseGraphStore, config=None):
        self.graph_store = graph_store
        self.config = config

    # ── Source lifecycle ─────────────────────────────────────────────────

    def confirm_source(self, doc_id: str):
        """Boost confidence of all triples from this source to 1.0."""
        self.graph_store.confirm_source(doc_id)

    def deprecate_source(self, doc_id: str):
        """Halve confidence of all triples from this source."""
        self.graph_store.deprecate_source(doc_id)

    def bulk_confirm(self, doc_ids: list[str]):
        for d in doc_ids:
            self.graph_store.confirm_source(d)

    def bulk_deprecate(self, doc_ids: list[str]):
        for d in doc_ids:
            self.graph_store.deprecate_source(d)

    # ── Conflict detection ───────────────────────────────────────────────

    def detect_conflicts(self, entity_name: str) -> list[dict]:
        """
        Find cases where the same (entity, relation_type) points to different
        targets from different sources — indicating potential contradictions.

        Uses Neo4j Cypher for Neo4j/Memgraph stores; returns empty list
        for stores without a Bolt driver.
        """
        if hasattr(self.graph_store, "driver") and hasattr(self.graph_store, "db"):
            return self._neo4j_detect_conflicts(entity_name)
        return []

    def _neo4j_detect_conflicts(self, entity_name: str) -> list[dict]:
        query = """
        MATCH (e:Entity)-[r]->(t:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
          AND NOT type(r) IN ['MENTIONED_IN', 'BELONGS_TO_COMMUNITY']
        RETURN e.name AS src, type(r) AS rel, t.name AS tgt,
               r.source_id AS source_id, r.confidence AS confidence
        ORDER BY rel, tgt
        """
        conflicts = []
        seen: dict[str, list[dict]] = {}
        with self.graph_store.driver.session(database=self.graph_store.db) as s:
            for rec in s.run(query, name=entity_name):
                key = f"{rec['src']}|{rec['rel']}"
                entry = dict(rec)
                if key in seen:
                    for existing in seen[key]:
                        if (existing["tgt"] != entry["tgt"]
                                and existing["source_id"] != entry["source_id"]):
                            conflicts.append({
                                "entity": rec["src"],
                                "relation": rec["rel"],
                                "versions": [existing, entry],
                            })
                    seen[key].append(entry)
                else:
                    seen[key] = [entry]
        return conflicts

    # ── Pruning ──────────────────────────────────────────────────────────

    def prune_low_confidence(self, threshold: float = None) -> int:
        t = threshold
        if t is None and self.config is not None:
            try:
                t = self.config.ingestion.confidence_threshold
            except AttributeError:
                t = 0.5
        if t is None:
            t = 0.5
        return self.graph_store.prune_low_confidence(t)

    # ── Reporting ────────────────────────────────────────────────────────

    def get_source_stats(self) -> list[dict]:
        """Per-document statistics. Full stats require Neo4j/Memgraph."""
        if hasattr(self.graph_store, "driver") and hasattr(self.graph_store, "db"):
            return self._neo4j_source_stats()
        return []

    def _neo4j_source_stats(self) -> list[dict]:
        query = """
        MATCH (d:DocumentMeta)
        OPTIONAL MATCH ()-[r {source_id: d.doc_id}]->()
        WITH d, count(r) AS triples
        RETURN d.doc_id AS doc_id, d.source_name AS source,
               d.status AS status, triples,
               d.chunk_count AS chunks, d.entity_count AS entities,
               d.relation_count AS relations
        ORDER BY d.updated_at DESC
        """
        with self.graph_store.driver.session(database=self.graph_store.db) as s:
            return [dict(r) for r in s.run(query)]

    def health_report(self) -> dict:
        return self.graph_store.health_report()
