"""
Memgraph Graph Store — open-source in-memory graph database.

Memgraph is Neo4j-compatible (Bolt protocol, Cypher queries) so this
implementation reuses most of the Neo4jStore logic, with adjustments for
Memgraph-specific behaviours (no GDS; uses MAGE algorithms instead).

Install: pip install gqlalchemy
Or use the Bolt driver directly: pip install neo4j
"""

from cognity_ai.stores.graph.neo4j import Neo4jStore
from cognity_ai.models.knowledge import Entity, Relation
from cognity_ai.models.retrieval import RetrievalResult, CommunityInfo


class MemgraphStore(Neo4jStore):
    """
    Memgraph graph store. Compatible with Neo4j Bolt protocol so it inherits
    Neo4jStore. Community detection uses Memgraph's built-in MAGE library
    (community_detection.get()) instead of GDS.
    """

    def __init__(self, memgraph_config, graphrag_config=None):
        # Accept either a MemgraphConfig or reuse Neo4jConfig fields
        from neo4j import GraphDatabase
        uri = getattr(memgraph_config, "bolt_uri",
                      getattr(memgraph_config, "uri", "bolt://localhost:7687"))
        user = getattr(memgraph_config, "user", "")
        password = getattr(memgraph_config, "password", "")
        database = getattr(memgraph_config, "database", "memgraph")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db = database
        self.grag = graphrag_config
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Memgraph uses slightly different index syntax."""
        queries = [
            "CREATE INDEX ON :Entity(name);",
            "CREATE INDEX ON :DocumentMeta(doc_id);",
            "CREATE INDEX ON :Community(community_id);",
            "CREATE INDEX ON :ChunkRef(chunk_id);",
        ]
        with self.driver.session(database=self.db) as s:
            for q in queries:
                try:
                    s.run(q)
                except Exception:
                    pass

    def detect_communities(self) -> list[dict]:
        """Use Memgraph's MAGE community detection (Louvain)."""
        with self.driver.session(database=self.db) as s:
            try:
                # MAGE community detection
                result = s.run("""
                    CALL community_detection.get()
                    YIELD node, community_id
                    WITH community_id, collect(node.name) AS members
                    RETURN community_id, members, size(members) AS size
                    ORDER BY size DESC
                """)
                communities = [dict(r) for r in result]
                if communities:
                    return communities
            except Exception:
                pass

            # Fallback: no community detection available
            return []
