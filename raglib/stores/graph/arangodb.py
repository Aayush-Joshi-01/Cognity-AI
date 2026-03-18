"""
ArangoDB Graph Store — multi-model graph database.

Uses AQL (ArangoDB Query Language) for graph traversal.
Install: pip install pyarango  OR  pip install python-arango
"""
from __future__ import annotations
from raglib.stores.graph.base import BaseGraphStore
from raglib.models.knowledge import Entity, Relation
from raglib.models.retrieval import RetrievalResult, CommunityInfo


class ArangoDBStore(BaseGraphStore):
    """
    ArangoDB-backed graph store.

    Uses python-arango client. Entity and Relation nodes stored in document
    collections; edges in an edge collection. Traversal via AQL.
    """

    def __init__(self, arangodb_config):
        try:
            from arango import ArangoClient
        except ImportError:
            raise ImportError("python-arango not installed. Run: pip install python-arango")

        from arango import ArangoClient

        url = getattr(arangodb_config, "url", "http://localhost:8529")
        username = getattr(arangodb_config, "username", "root")
        password = getattr(arangodb_config, "password", "")
        db_name = getattr(arangodb_config, "database", "raglib")

        client = ArangoClient(hosts=url)
        sys_db = client.db("_system", username=username, password=password)

        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)

        self._db = client.db(db_name, username=username, password=password)
        self._ensure_collections()

        self._doc_meta: dict[str, dict] = {}  # in-memory for lifecycle

    def _ensure_collections(self):
        for col in ["entities", "communities"]:
            if not self._db.has_collection(col):
                self._db.create_collection(col)
        for col in ["relations", "chunk_entity"]:
            if not self._db.has_collection(col):
                self._db.create_collection(col, edge=True)

        # Create graph if not exists
        if not self._db.has_graph("rag_graph"):
            self._db.create_graph("rag_graph", edge_definitions=[{
                "edge_collection": "relations",
                "from_vertex_collections": ["entities"],
                "to_vertex_collections": ["entities"],
            }])

    def _entity_key(self, name: str) -> str:
        return name.replace(" ", "_").replace("/", "_")[:100]

    def upsert_entity(self, entity: Entity):
        col = self._db.collection("entities")
        key = self._entity_key(entity.name)
        doc = entity.model_dump()
        doc["_key"] = key
        try:
            existing = col.get(key)
            if existing:
                if entity.confidence > existing.get("confidence", 0):
                    doc["mentions"] = existing.get("mentions", 1) + entity.mentions
                    col.update(doc)
                else:
                    col.update({"_key": key, "mentions": existing.get("mentions", 1) + entity.mentions})
            else:
                col.insert(doc)
        except Exception:
            try:
                col.insert(doc)
            except Exception:
                pass

    def upsert_relation(self, relation: Relation):
        col = self._db.collection("relations")
        src_key = self._entity_key(relation.source_entity)
        tgt_key = self._entity_key(relation.target_entity)
        edge_key = f"{src_key}_{relation.relation_type}_{tgt_key}"[:100]
        doc = relation.model_dump()
        doc["_key"] = edge_key
        doc["_from"] = f"entities/{src_key}"
        doc["_to"] = f"entities/{tgt_key}"
        try:
            existing = col.get(edge_key)
            if existing:
                col.update({"_key": edge_key, "weight": existing.get("weight", 1) + relation.weight})
            else:
                col.insert(doc)
        except Exception:
            try:
                col.insert(doc)
            except Exception:
                pass

    def link_chunk_to_entities(self, chunk_id: str, doc_id: str, entity_names: list[str]):
        col = self._db.collection("chunk_entity")
        for name in entity_names:
            key = self._entity_key(name)
            edge_key = f"{chunk_id}_{key}"[:100]
            doc = {
                "_key": edge_key,
                "_from": f"entities/{key}",
                "_to": f"entities/{key}",  # placeholder; chunk refs not a vertex collection
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "entity_name": name,
            }
            try:
                col.insert(doc)
            except Exception:
                pass

    def upsert_doc_meta(self, doc_id: str, content_hash: str, source_name: str,
                        status: str = "pending", stats: dict = None):
        self._doc_meta[doc_id] = {
            "doc_id": doc_id, "content_hash": content_hash,
            "source_name": source_name, "status": status,
            **(stats or {}),
        }

    def remove_doc_subgraph(self, doc_id: str):
        try:
            self._db.aql.execute(
                "FOR r IN relations FILTER r.source_id == @doc_id REMOVE r IN relations",
                bind_vars={"doc_id": doc_id},
            )
        except Exception:
            pass
        self._doc_meta.pop(doc_id, None)

    def retrieve_subgraph(self, entity_names: list[str], hops: int = 2,
                          limit: int = 20) -> list[RetrievalResult]:
        results = []
        for name in entity_names[:3]:
            key = self._entity_key(name)
            try:
                cursor = self._db.aql.execute(
                    f"""
                    FOR v, e, p IN 1..{hops} ANY 'entities/{key}' relations
                        LIMIT {limit}
                        RETURN {{src: p.vertices[0].name, rel: e.relation_type,
                                 tgt: v.name, conf: e.confidence, desc: e.description,
                                 source_id: e.source_id}}
                    """
                )
                for rec in cursor:
                    triple = f"{rec['src']} --[{rec['rel']}]--> {rec['tgt']}"
                    if rec.get("desc"):
                        triple += f" ({rec['desc']})"
                    results.append(RetrievalResult(
                        content=triple,
                        score=rec.get("conf") or 0.5,
                        source="graph",
                        metadata={"src": rec["src"], "rel": rec["rel"],
                                  "tgt": rec["tgt"], "source_id": rec.get("source_id", "")},
                    ))
            except Exception:
                pass
        return results[:limit]

    def retrieve_entity_context(self, entity_name: str) -> list[RetrievalResult]:
        key = self._entity_key(entity_name)
        try:
            col = self._db.collection("entities")
            entity = col.get(key)
            if not entity:
                return []
            parts = [f"Entity: {entity.get('name', entity_name)} ({entity.get('entity_type', 'Unknown')})"]
            if entity.get("description"):
                parts.append(f"Description: {entity['description']}")
            return [RetrievalResult(content="\n".join(parts), score=1.0, source="graph",
                                    metadata={"entity": entity_name})]
        except Exception:
            return []

    def global_community_search(self, top_n: int = 5) -> list[RetrievalResult]:
        results = []
        try:
            cursor = self._db.aql.execute(
                f"FOR c IN communities SORT c.rank DESC LIMIT {top_n} RETURN c"
            )
            for rec in cursor:
                results.append(RetrievalResult(
                    content=f"[{rec.get('title', rec.get('community_id', '?'))}] {rec.get('summary', '')}",
                    score=rec.get("rank", 0.5),
                    source="community",
                    metadata={"community_id": rec.get("community_id", "")},
                ))
        except Exception:
            pass
        return results

    def get_chunks_for_entities(self, entity_names: list[str]) -> list[str]:
        chunk_ids = []
        try:
            for name in entity_names:
                cursor = self._db.aql.execute(
                    "FOR ce IN chunk_entity FILTER ce.entity_name == @name RETURN ce.chunk_id",
                    bind_vars={"name": name},
                )
                chunk_ids.extend(list(cursor))
        except Exception:
            pass
        return list(dict.fromkeys(chunk_ids))

    def detect_communities(self) -> list[dict]:
        """ArangoDB doesn't have built-in Leiden; return empty (use NetworkX fallback)."""
        return []

    def get_community_entities(self, community_id) -> list[dict]:
        return []

    def store_community_summary(self, community: CommunityInfo):
        try:
            col = self._db.collection("communities")
            key = str(community.community_id).replace("-", "_")
            doc = community.model_dump(exclude={"embedding"})
            doc["_key"] = key
            try:
                col.insert(doc)
            except Exception:
                col.update({"_key": key, **doc})
        except Exception:
            pass

    def confirm_source(self, doc_id: str):
        if doc_id in self._doc_meta:
            self._doc_meta[doc_id]["status"] = "confirmed"

    def deprecate_source(self, doc_id: str):
        if doc_id in self._doc_meta:
            self._doc_meta[doc_id]["status"] = "deprecated"

    def get_doc_status(self, doc_id: str) -> str | None:
        return self._doc_meta.get(doc_id, {}).get("status")

    def prune_low_confidence(self, threshold: float = 0.5) -> int:
        try:
            cursor = self._db.aql.execute(
                "FOR r IN relations FILTER r.confidence < @t REMOVE r IN relations RETURN 1",
                bind_vars={"t": threshold},
            )
            return sum(1 for _ in cursor)
        except Exception:
            return 0

    def health_report(self) -> dict:
        try:
            entities = self._db.collection("entities").count()
            relations = self._db.collection("relations").count()
            communities = self._db.collection("communities").count()
        except Exception:
            entities = relations = communities = 0
        confirmed = sum(1 for m in self._doc_meta.values() if m.get("status") == "confirmed")
        return {
            "entities": entities, "relations": relations,
            "documents": len(self._doc_meta), "confirmed_sources": confirmed,
            "communities": communities, "avg_confidence": 0.0,
        }
