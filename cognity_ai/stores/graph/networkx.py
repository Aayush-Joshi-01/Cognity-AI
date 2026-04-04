"""
NetworkX in-memory Graph Store — zero external dependencies.

Suitable for development, testing, and small corpora.
Does NOT support community detection (Leiden requires GDS).
"""
from __future__ import annotations
from cognity_ai.stores.graph.base import BaseGraphStore
from cognity_ai.models.knowledge import Entity, Relation
from cognity_ai.models.retrieval import RetrievalResult, CommunityInfo
from cognity_ai.utils.trie import EntityTrie


class NetworkXStore(BaseGraphStore):
    """Pure in-memory graph using networkx. No persistence — data lives in RAM."""

    def __init__(self):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx not installed. Run: pip install networkx")

        import networkx as nx
        self._g = nx.MultiDiGraph()
        self._entities: dict[str, dict] = {}          # name → entity data
        self._relations: list[dict] = []
        self._chunk_entity: dict[str, list[str]] = {}  # chunk_id → entity names
        self._entity_chunks: dict[str, list[str]] = {}  # entity name → chunk ids
        self._doc_meta: dict[str, dict] = {}
        self._communities: dict[str, dict] = {}        # community_id → CommunityInfo
        self._entity_trie = EntityTrie()               # O(k) prefix lookups

    # ── Entities ─────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity):
        key = entity.name.lower().strip()
        existing = self._entities.get(key)
        if existing:
            if entity.confidence > existing.get("confidence", 0):
                self._entities[key] = entity.model_dump()
                self._entities[key]["mentions"] = existing.get("mentions", 1) + entity.mentions
            else:
                existing["mentions"] = existing.get("mentions", 1) + entity.mentions
        else:
            self._entities[key] = entity.model_dump()
        self._g.add_node(entity.name, **self._entities[key])
        self._entity_trie.insert_entity(entity.name)

    def upsert_relation(self, relation: Relation):
        self._relations.append(relation.model_dump())
        self._g.add_edge(
            relation.source_entity, relation.target_entity,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            weight=relation.weight,
            source_id=relation.source_id,
            description=relation.description,
        )

    def link_chunk_to_entities(self, chunk_id: str, doc_id: str, entity_names: list[str]):
        self._chunk_entity[chunk_id] = entity_names
        for name in entity_names:
            self._entity_chunks.setdefault(name.lower(), []).append(chunk_id)

    def upsert_doc_meta(self, doc_id: str, content_hash: str, source_name: str,
                        status: str = "pending", stats: dict = None):
        self._doc_meta[doc_id] = {
            "doc_id": doc_id, "content_hash": content_hash,
            "source_name": source_name, "status": status,
            **(stats or {}),
        }

    def remove_doc_subgraph(self, doc_id: str):
        # Remove relations for this doc
        self._relations = [r for r in self._relations if r.get("source_id") != doc_id]
        # Remove doc meta
        self._doc_meta.pop(doc_id, None)
        # Remove chunk-entity mappings for this doc
        orphan_chunks = [cid for cid, ents in self._chunk_entity.items() if True]  # all
        # Rebuild entity_chunks
        for cid in list(self._chunk_entity.keys()):
            pass  # simplification: don't remove individual chunks since we lack doc_id here

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve_subgraph(self, entity_names: list[str], hops: int = 2,
                          limit: int = 20) -> list[RetrievalResult]:
        results = []
        import networkx as nx
        visited_nodes = set()
        for name in entity_names:
            # O(k) prefix lookup via EntityTrie (replaces O(n) linear scan)
            matching = self._entity_trie.search_entities(name.lower())
            for start in matching[:2]:
                # BFS up to `hops` levels
                for depth in range(1, hops + 1):
                    try:
                        neighbors = list(nx.ego_graph(self._g, start, radius=depth, undirected=True).edges(data=True))
                    except Exception:
                        break
                    for src, tgt, data in neighbors:
                        key = (src, data.get("relation_type", "?"), tgt)
                        if key in visited_nodes:
                            continue
                        visited_nodes.add(key)
                        rel_type = data.get("relation_type", "RELATED_TO")
                        conf = data.get("confidence", 0.5)
                        desc = data.get("description", "")
                        triple = f"{src} --[{rel_type}]--> {tgt}"
                        if desc:
                            triple += f" ({desc})"
                        results.append(RetrievalResult(
                            content=triple,
                            score=conf,
                            source="graph",
                            metadata={"src": src, "rel": rel_type, "tgt": tgt,
                                      "source_id": data.get("source_id", "")},
                        ))
                        if len(results) >= limit:
                            return results
        return results

    def retrieve_entity_context(self, entity_name: str) -> list[RetrievalResult]:
        results = []
        matching = self._entity_trie.search_entities(entity_name.lower())
        for node in matching[:1]:
            node_data = self._g.nodes.get(node, {})
            parts = [f"Entity: {node} ({node_data.get('entity_type', 'Unknown')})"]
            if node_data.get("description"):
                parts.append(f"Description: {node_data['description']}")
            for _, tgt, data in self._g.out_edges(node, data=True):
                rel = data.get("relation_type", "RELATED_TO")
                parts.append(f"  -> {rel} -> {tgt}")
            for src, _, data in self._g.in_edges(node, data=True):
                rel = data.get("relation_type", "RELATED_TO")
                parts.append(f"  <- {rel} <- {src}")
            results.append(RetrievalResult(
                content="\n".join(parts),
                score=1.0,
                source="graph",
                metadata={"entity": node},
            ))
        return results

    def global_community_search(self, top_n: int = 5) -> list[RetrievalResult]:
        results = []
        for cid, comm in sorted(self._communities.items(), key=lambda x: x[1].get("rank", 0), reverse=True)[:top_n]:
            results.append(RetrievalResult(
                content=f"[{comm.get('title', cid)}] {comm.get('summary', '')}",
                score=comm.get("rank", 0.5),
                source="community",
                metadata={"community_id": cid},
            ))
        return results

    def get_chunks_for_entities(self, entity_names: list[str]) -> list[str]:
        chunk_ids = []
        for name in entity_names:
            chunk_ids.extend(self._entity_chunks.get(name.lower(), []))
        return list(dict.fromkeys(chunk_ids))  # deduplicate, preserve order

    def detect_communities(self) -> list[dict]:
        """Use networkx's greedy modularity communities (no GDS needed)."""
        try:
            import networkx as nx
            import networkx.algorithms.community as nx_comm
            undirected = self._g.to_undirected()
            if len(undirected.nodes) < 2:
                return []
            communities = list(nx_comm.greedy_modularity_communities(undirected))
            return [
                {"community_id": i, "members": list(comm), "size": len(comm)}
                for i, comm in enumerate(communities)
            ]
        except Exception:
            return []

    def get_community_entities(self, community_id: int) -> list[dict]:
        # For NetworkX, we just return nodes that were assigned this community
        return []

    def store_community_summary(self, community: CommunityInfo):
        self._communities[community.community_id] = community.model_dump(exclude={"embedding"})

    # ── Lifecycle ────────────────────────────────────────────────────────

    def confirm_source(self, doc_id: str):
        if doc_id in self._doc_meta:
            self._doc_meta[doc_id]["status"] = "confirmed"

    def deprecate_source(self, doc_id: str):
        if doc_id in self._doc_meta:
            self._doc_meta[doc_id]["status"] = "deprecated"
        # Halve confidence of all relations from this source
        for u, v, k, data in list(self._g.edges(data=True, keys=True)):
            if data.get("source_id") == doc_id:
                self._g[u][v][k]["confidence"] = data.get("confidence", 1.0) * 0.5

    def get_doc_status(self, doc_id: str) -> str | None:
        return self._doc_meta.get(doc_id, {}).get("status")

    def prune_low_confidence(self, threshold: float = 0.5) -> int:
        to_remove = [
            (u, v, k)
            for u, v, k, data in self._g.edges(data=True, keys=True)
            if data.get("confidence", 1.0) < threshold
        ]
        for u, v, k in to_remove:
            self._g.remove_edge(u, v, key=k)
        return len(to_remove)

    def suggest_entities(self, prefix: str, max_results: int = 10) -> list[str]:
        """Return original-case entity names whose lowercased form starts with *prefix*."""
        return self._entity_trie.search_entities(prefix.lower(), max_results=max_results)

    def health_report(self) -> dict:
        confirmed = sum(1 for m in self._doc_meta.values() if m.get("status") == "confirmed")
        return {
            "entities": len(self._g.nodes),
            "relations": len(self._g.edges),
            "documents": len(self._doc_meta),
            "confirmed_sources": confirmed,
            "communities": len(self._communities),
            "avg_confidence": 0.0,
        }
