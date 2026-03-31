"""
Neo4j Graph Store — direct migration of hybrid_rag/graph_manager.py.

Implements BaseGraphStore against a Neo4j 5.x instance with optional
GDS (Leiden/Louvain community detection) and APOC (subgraph traversal).
"""

from cognity_ai.stores.graph.base import BaseGraphStore
from cognity_ai.models.knowledge import Entity, Relation
from cognity_ai.models.retrieval import RetrievalResult, CommunityInfo


class Neo4jStore(BaseGraphStore):
    def __init__(self, neo4j_config, graphrag_config=None):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.user, neo4j_config.password),
        )
        self.db = neo4j_config.database
        self.grag = graphrag_config
        self._ensure_indexes()

    def _ensure_indexes(self):
        queries = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_source IF NOT EXISTS FOR (e:Entity) ON (e.source_id)",
            "CREATE INDEX doc_meta IF NOT EXISTS FOR (d:DocumentMeta) ON (d.doc_id)",
            "CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.community_id)",
            "CREATE INDEX chunk_ref IF NOT EXISTS FOR (ch:ChunkRef) ON (ch.chunk_id)",
        ]
        with self.driver.session(database=self.db) as s:
            for q in queries:
                try:
                    s.run(q)
                except Exception:
                    pass

    def close(self):
        self.driver.close()

    # ── Entity Upsert ────────────────────────────────────────────────────

    def upsert_entity(self, entity: Entity):
        label = entity.entity_type.replace(" ", "")
        query = f"""
        MERGE (e:Entity {{name: $name}})
        ON CREATE SET
            e.entity_type = $entity_type,
            e.description = $description,
            e.properties = $properties,
            e.source_id = $source_id,
            e.confidence = $confidence,
            e.extraction_method = $extraction_method,
            e.mentions = $mentions,
            e.created_at = datetime()
        ON MATCH SET
            e.entity_type = CASE WHEN $confidence > e.confidence
                                 THEN $entity_type ELSE e.entity_type END,
            e.description = CASE WHEN size($description) > size(coalesce(e.description, ''))
                                 THEN $description ELSE e.description END,
            e.confidence = CASE WHEN $confidence > e.confidence
                                THEN $confidence ELSE e.confidence END,
            e.mentions = e.mentions + $mentions,
            e.updated_at = datetime()
        SET e:{label}
        """
        with self.driver.session(database=self.db) as s:
            s.run(query, **entity.model_dump())

    # ── Relation Upsert ──────────────────────────────────────────────────

    def upsert_relation(self, relation: Relation):
        query = """
        MATCH (a:Entity {name: $source_entity})
        MATCH (b:Entity {name: $target_entity})
        MERGE (a)-[r:%s]->(b)
        ON CREATE SET
            r.description = $description,
            r.properties = $properties,
            r.source_id = $source_id,
            r.confidence = $confidence,
            r.weight = $weight,
            r.extraction_method = $extraction_method,
            r.created_at = datetime()
        ON MATCH SET
            r.description = CASE WHEN size($description) > size(coalesce(r.description, ''))
                                 THEN $description ELSE r.description END,
            r.confidence = CASE WHEN $confidence > r.confidence
                                THEN $confidence ELSE r.confidence END,
            r.weight = r.weight + $weight,
            r.updated_at = datetime()
        """ % relation.relation_type
        with self.driver.session(database=self.db) as s:
            s.run(query, **relation.model_dump())

    # ── Chunk-Entity linking ─────────────────────────────────────────────

    def link_chunk_to_entities(self, chunk_id: str, doc_id: str, entity_names: list[str]):
        with self.driver.session(database=self.db) as s:
            s.run(
                "MERGE (ch:ChunkRef {chunk_id: $chunk_id}) SET ch.doc_id = $doc_id",
                chunk_id=chunk_id, doc_id=doc_id,
            )
            for name in entity_names:
                s.run(
                    """
                    MATCH (e:Entity {name: $name})
                    MATCH (ch:ChunkRef {chunk_id: $chunk_id})
                    MERGE (e)-[:MENTIONED_IN]->(ch)
                    """,
                    name=name, chunk_id=chunk_id,
                )

    # ── Document Meta ────────────────────────────────────────────────────

    def upsert_doc_meta(self, doc_id: str, content_hash: str, source_name: str,
                        status: str = "pending", stats: dict = None):
        with self.driver.session(database=self.db) as s:
            s.run(
                """
                MERGE (d:DocumentMeta {doc_id: $doc_id})
                SET d.content_hash = $content_hash,
                    d.source_name = $source_name,
                    d.status = $status,
                    d.chunk_count = $chunk_count,
                    d.entity_count = $entity_count,
                    d.relation_count = $relation_count,
                    d.updated_at = datetime()
                ON CREATE SET d.ingested_at = datetime()
                """,
                doc_id=doc_id, content_hash=content_hash, source_name=source_name,
                status=status,
                chunk_count=(stats or {}).get("chunks", 0),
                entity_count=(stats or {}).get("entities", 0),
                relation_count=(stats or {}).get("relations", 0),
            )

    # ── Removal ──────────────────────────────────────────────────────────

    def remove_doc_subgraph(self, doc_id: str):
        with self.driver.session(database=self.db) as s:
            s.run("MATCH (ch:ChunkRef {doc_id: $doc_id}) DETACH DELETE ch", doc_id=doc_id)
            s.run("MATCH ()-[r]->() WHERE r.source_id = $doc_id DELETE r", doc_id=doc_id)
            s.run(
                "MATCH (e:Entity {source_id: $doc_id}) WHERE NOT (e)--() DELETE e",
                doc_id=doc_id,
            )
            s.run("MATCH (d:DocumentMeta {doc_id: $doc_id}) DELETE d", doc_id=doc_id)

    # ── Community Detection ──────────────────────────────────────────────

    def detect_communities(self) -> list[dict]:
        with self.driver.session(database=self.db) as s:
            try:
                s.run("CALL gds.graph.drop('rag_graph', false)")
            except Exception:
                pass

            s.run("""
                CALL gds.graph.project(
                    'rag_graph', 'Entity',
                    {_ALL_: {type: '*', orientation: 'UNDIRECTED', properties: ['weight', 'confidence']}}
                )
            """)

            max_levels = 3
            resolution = 1.0
            if self.grag:
                max_levels = getattr(self.grag, "max_community_levels", 3)
                resolution = getattr(self.grag, "leiden_resolution", 1.0)

            try:
                result = s.run("""
                    CALL gds.leiden.write('rag_graph', {
                        writeProperty: 'community_level_0',
                        maxLevels: $max_levels,
                        gamma: $resolution,
                        includeIntermediateCommunities: true,
                        intermediateCommunitiesWriteProperty: 'community_levels'
                    })
                    YIELD communityCount, modularity
                    RETURN communityCount, modularity
                """, max_levels=max_levels, resolution=resolution)
            except Exception:
                result = s.run("""
                    CALL gds.louvain.write('rag_graph', {
                        writeProperty: 'community_level_0',
                        includeIntermediateCommunities: true,
                        intermediateCommunitiesWriteProperty: 'community_levels'
                    })
                    YIELD communityCount, modularity
                    RETURN communityCount, modularity
                """)

            result.single()

            try:
                s.run("CALL gds.graph.drop('rag_graph', false)")
            except Exception:
                pass

            communities = s.run("""
                MATCH (e:Entity)
                WHERE e.community_level_0 IS NOT NULL
                RETURN e.community_level_0 AS community_id,
                       collect(e.name) AS members,
                       count(e) AS size
                ORDER BY size DESC
            """)
            return [dict(r) for r in communities]

    def get_community_entities(self, community_id: int) -> list[dict]:
        with self.driver.session(database=self.db) as s:
            result = s.run("""
                MATCH (e:Entity {community_level_0: $cid})
                OPTIONAL MATCH (e)-[r]->(t:Entity {community_level_0: $cid})
                RETURN e.name AS entity, e.entity_type AS type,
                       e.description AS description,
                       type(r) AS rel, t.name AS target,
                       r.description AS rel_desc
            """, cid=community_id)
            return [dict(r) for r in result]

    def store_community_summary(self, community: CommunityInfo):
        with self.driver.session(database=self.db) as s:
            s.run("""
                MERGE (c:Community {community_id: $community_id})
                SET c.level = $level,
                    c.title = $title,
                    c.summary = $summary,
                    c.rank = $rank,
                    c.entity_count = size($entity_names),
                    c.updated_at = datetime()
            """, **community.model_dump(exclude={"embedding", "entity_names", "parent_community"}),
                 entity_names=community.entity_names)

            for name in community.entity_names:
                s.run("""
                    MATCH (e:Entity {name: $name})
                    MATCH (c:Community {community_id: $cid})
                    MERGE (e)-[:BELONGS_TO_COMMUNITY]->(c)
                """, name=name, cid=community.community_id)

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve_subgraph(self, entity_names: list[str], hops: int = 2,
                          limit: int = 20) -> list[RetrievalResult]:
        query = """
        UNWIND $names AS seed
        MATCH (start:Entity)
        WHERE toLower(start.name) CONTAINS toLower(seed)
        CALL apoc.path.subgraphAll(start, {maxLevel: $hops, limit: $limit})
        YIELD relationships
        UNWIND relationships AS r
        WITH startNode(r) AS a, type(r) AS rel, endNode(r) AS b, r
        WHERE rel <> 'MENTIONED_IN' AND rel <> 'BELONGS_TO_COMMUNITY'
        RETURN DISTINCT a.name AS src, rel, b.name AS tgt,
               a.entity_type AS src_type, b.entity_type AS tgt_type,
               r.confidence AS confidence, r.source_id AS source_id,
               r.description AS rel_desc,
               a.description AS src_desc, b.description AS tgt_desc
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session(database=self.db) as s:
            try:
                records = s.run(query, names=entity_names, hops=hops, limit=limit)
                for rec in records:
                    desc = rec["rel_desc"] or ""
                    triple = f"{rec['src']} --[{rec['rel']}]--> {rec['tgt']}"
                    if desc:
                        triple += f" ({desc})"
                    results.append(RetrievalResult(
                        content=triple,
                        score=rec["confidence"] or 0.5,
                        source="graph",
                        metadata={
                            "src": rec["src"], "rel": rec["rel"], "tgt": rec["tgt"],
                            "source_id": rec["source_id"],
                            "src_desc": rec["src_desc"] or "",
                            "tgt_desc": rec["tgt_desc"] or "",
                        },
                    ))
            except Exception:
                # APOC not available — fall back to simple Cypher
                results = self._retrieve_subgraph_fallback(entity_names, hops, limit)
        return results

    def _retrieve_subgraph_fallback(self, entity_names: list[str], hops: int,
                                    limit: int) -> list[RetrievalResult]:
        """Fallback subgraph retrieval without APOC."""
        query = """
        UNWIND $names AS seed
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower(seed)
          AND NOT type(r) IN ['MENTIONED_IN', 'BELONGS_TO_COMMUNITY']
        RETURN DISTINCT a.name AS src, type(r) AS rel, b.name AS tgt,
               r.confidence AS confidence, r.source_id AS source_id,
               r.description AS rel_desc,
               a.description AS src_desc, b.description AS tgt_desc
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session(database=self.db) as s:
            for rec in s.run(query, names=entity_names, limit=limit):
                desc = rec["rel_desc"] or ""
                triple = f"{rec['src']} --[{rec['rel']}]--> {rec['tgt']}"
                if desc:
                    triple += f" ({desc})"
                results.append(RetrievalResult(
                    content=triple,
                    score=rec["confidence"] or 0.5,
                    source="graph",
                    metadata={
                        "src": rec["src"], "rel": rec["rel"], "tgt": rec["tgt"],
                        "source_id": rec["source_id"],
                        "src_desc": rec["src_desc"] or "",
                        "tgt_desc": rec["tgt_desc"] or "",
                    },
                ))
        return results

    def retrieve_entity_context(self, entity_name: str) -> list[RetrievalResult]:
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)-[r]->(t)
        WHERE NOT type(r) IN ['MENTIONED_IN', 'BELONGS_TO_COMMUNITY']
        OPTIONAL MATCH (s)-[r2]->(e)
        WHERE NOT type(r2) IN ['MENTIONED_IN', 'BELONGS_TO_COMMUNITY']
        WITH e,
             collect(DISTINCT {rel: type(r), tgt: t.name, desc: r.description, conf: r.confidence}) AS out,
             collect(DISTINCT {rel: type(r2), src: s.name, desc: r2.description, conf: r2.confidence}) AS inc
        RETURN e.name AS name, e.entity_type AS etype,
               e.description AS desc, e.mentions AS mentions, out, inc
        """
        results = []
        with self.driver.session(database=self.db) as s:
            for rec in s.run(query, name=entity_name):
                parts = [f"Entity: {rec['name']} ({rec['etype']})"]
                if rec["desc"]:
                    parts.append(f"Description: {rec['desc']}")
                for o in rec["out"]:
                    if o["tgt"]:
                        line = f"  -> {o['rel']} -> {o['tgt']}"
                        if o["desc"]:
                            line += f" ({o['desc']})"
                        parts.append(line)
                for i in rec["inc"]:
                    if i["src"]:
                        line = f"  <- {i['rel']} <- {i['src']}"
                        if i["desc"]:
                            line += f" ({i['desc']})"
                        parts.append(line)
                results.append(RetrievalResult(
                    content="\n".join(parts),
                    score=1.0,
                    source="graph",
                    metadata={"entity": rec["name"], "mentions": rec["mentions"]},
                ))
        return results

    def global_community_search(self, top_n: int = 5) -> list[RetrievalResult]:
        query = """
        MATCH (c:Community)
        RETURN c.community_id AS cid, c.title AS title,
               c.summary AS summary, c.rank AS rank,
               c.entity_count AS entity_count
        ORDER BY c.rank DESC
        LIMIT $top_n
        """
        results = []
        with self.driver.session(database=self.db) as s:
            for rec in s.run(query, top_n=top_n):
                results.append(RetrievalResult(
                    content=f"[{rec['title']}] {rec['summary']}",
                    score=rec["rank"] or 0.5,
                    source="community",
                    metadata={"community_id": rec["cid"], "entity_count": rec["entity_count"]},
                ))
        return results

    def get_chunks_for_entities(self, entity_names: list[str]) -> list[str]:
        query = """
        UNWIND $names AS name
        MATCH (e:Entity)-[:MENTIONED_IN]->(ch:ChunkRef)
        WHERE toLower(e.name) CONTAINS toLower(name)
        RETURN DISTINCT ch.chunk_id AS chunk_id
        """
        with self.driver.session(database=self.db) as s:
            return [r["chunk_id"] for r in s.run(query, names=entity_names)]

    # ── Lifecycle ────────────────────────────────────────────────────────

    def confirm_source(self, doc_id: str):
        with self.driver.session(database=self.db) as s:
            s.run("MATCH (d:DocumentMeta {doc_id: $d}) SET d.status = 'confirmed'", d=doc_id)
            s.run("""
                MATCH ()-[r]->() WHERE r.source_id = $d
                SET r.confidence = CASE WHEN r.confidence < 1.0 THEN 1.0 ELSE r.confidence END
            """, d=doc_id)
            s.run("MATCH (e:Entity {source_id: $d}) SET e.confidence = 1.0", d=doc_id)

    def deprecate_source(self, doc_id: str):
        with self.driver.session(database=self.db) as s:
            s.run("MATCH (d:DocumentMeta {doc_id: $d}) SET d.status = 'deprecated'", d=doc_id)
            s.run(
                "MATCH ()-[r]->() WHERE r.source_id = $d SET r.confidence = r.confidence * 0.5",
                d=doc_id,
            )

    def get_doc_status(self, doc_id: str) -> str | None:
        with self.driver.session(database=self.db) as s:
            result = s.run(
                "MATCH (d:DocumentMeta {doc_id: $d}) RETURN d.status AS s", d=doc_id
            ).single()
            return result["s"] if result else None

    def prune_low_confidence(self, threshold: float = 0.5) -> int:
        with self.driver.session(database=self.db) as s:
            result = s.run(
                "MATCH ()-[r]->() WHERE r.confidence < $t DELETE r RETURN count(*) AS n",
                t=threshold,
            )
            return result.single()["n"]

    def health_report(self) -> dict:
        with self.driver.session(database=self.db) as s:
            entity_count = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
            rel_count = s.run(
                "MATCH ()-[r]->() WHERE NOT type(r) IN ['MENTIONED_IN','BELONGS_TO_COMMUNITY'] RETURN count(r) AS n"
            ).single()["n"]
            doc_count = s.run("MATCH (d:DocumentMeta) RETURN count(d) AS n").single()["n"]
            confirmed = s.run(
                "MATCH (d:DocumentMeta {status:'confirmed'}) RETURN count(d) AS n"
            ).single()["n"]
            community_count = s.run("MATCH (c:Community) RETURN count(c) AS n").single()["n"]
            avg_conf_rec = s.run(
                "MATCH ()-[r]->() WHERE r.confidence IS NOT NULL RETURN avg(r.confidence) AS a"
            ).single()
            avg_conf = avg_conf_rec["a"] if avg_conf_rec else 0
        return {
            "entities": entity_count,
            "relations": rel_count,
            "documents": doc_count,
            "confirmed_sources": confirmed,
            "communities": community_count,
            "avg_confidence": round(avg_conf or 0, 3),
        }
