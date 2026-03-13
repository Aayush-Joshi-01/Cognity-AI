"""
Knowledge lifecycle manager:
  - Confirm/deprecate sources → confidence propagation
  - Conflict detection across sources
  - Low-confidence pruning
  - Source stats and health reporting
"""

from graph_manager import GraphManager
from config import Config


class KnowledgeUpdater:
    def __init__(self, graph: GraphManager, config: Config):
        self.graph = graph
        self.config = config

    def confirm_source(self, doc_id: str):
        self.graph.confirm_source(doc_id)

    def deprecate_source(self, doc_id: str):
        self.graph.deprecate_source(doc_id)

    def bulk_confirm(self, doc_ids: list[str]):
        for d in doc_ids:
            self.graph.confirm_source(d)

    def bulk_deprecate(self, doc_ids: list[str]):
        for d in doc_ids:
            self.graph.deprecate_source(d)

    def detect_conflicts(self, entity_name: str) -> list[dict]:
        """Same (entity, relation_type) pointing to different targets from different sources."""
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
        with self.graph.driver.session(database=self.graph.db) as s:
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

    def prune_low_confidence(self, threshold: float = None) -> int:
        t = threshold or self.config.ingestion.confidence_threshold
        return self.graph.prune_low_confidence(t)

    def get_source_stats(self) -> list[dict]:
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
        with self.graph.driver.session(database=self.graph.db) as s:
            return [dict(r) for r in s.run(query)]

    def health_report(self) -> dict:
        """Overall knowledge base health."""
        with self.graph.driver.session(database=self.graph.db) as s:
            entity_count = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
            rel_count = s.run("MATCH ()-[r]->() WHERE NOT type(r) IN ['MENTIONED_IN','BELONGS_TO_COMMUNITY'] RETURN count(r) AS n").single()["n"]
            doc_count = s.run("MATCH (d:DocumentMeta) RETURN count(d) AS n").single()["n"]
            confirmed = s.run("MATCH (d:DocumentMeta {status:'confirmed'}) RETURN count(d) AS n").single()["n"]
            community_count = s.run("MATCH (c:Community) RETURN count(c) AS n").single()["n"]
            avg_conf = s.run("MATCH ()-[r]->() WHERE r.confidence IS NOT NULL RETURN avg(r.confidence) AS a").single()["a"]

        return {
            "entities": entity_count,
            "relations": rel_count,
            "documents": doc_count,
            "confirmed_sources": confirmed,
            "communities": community_count,
            "avg_confidence": round(avg_conf or 0, 3),
        }
