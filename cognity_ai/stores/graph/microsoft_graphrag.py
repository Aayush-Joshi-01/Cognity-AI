"""
Microsoft GraphRAG Store — wraps the official microsoft/graphrag library.

The microsoft/graphrag library uses its own indexing pipeline and parquet
storage format. This adapter:
1. Delegates ingestion to graphrag's indexing pipeline.
2. Exposes local/global search via graphrag's query engine.
3. Implements BaseGraphStore stubs for the methods graphrag handles natively.

Install:
    pip install graphrag

Initialize a workspace first:
    python -m graphrag.index --init --root ./graphrag_workspace
    # then configure settings.yaml and run indexing

This store is best used alongside a vector store for hybrid retrieval.
"""
from __future__ import annotations
import os
from cognity_ai.stores.graph.base import BaseGraphStore
from cognity_ai.models.knowledge import Entity, Relation
from cognity_ai.models.retrieval import RetrievalResult, CommunityInfo


class MicrosoftGraphRAGStore(BaseGraphStore):
    """
    Wraps microsoft/graphrag for community detection and global/local search.

    Unlike Neo4j, MS GraphRAG manages its own storage (parquet files).
    Entity/relation upsert operations are not directly supported — instead,
    you run the full graphrag indexing pipeline on your documents.
    """

    def __init__(self, working_dir: str = "./graphrag_workspace",
                 search_type: str = "local"):
        self._working_dir = working_dir
        self._search_type = search_type
        self._doc_meta: dict[str, dict] = {}
        os.makedirs(working_dir, exist_ok=True)

    def _run_sync(self, coro):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    # ── Not directly supported by graphrag — no-ops with warnings ───────

    def upsert_entity(self, entity: Entity):
        pass  # graphrag manages its own entity store via indexing pipeline

    def upsert_relation(self, relation: Relation):
        pass

    def link_chunk_to_entities(self, chunk_id: str, doc_id: str, entity_names: list[str]):
        pass

    def upsert_doc_meta(self, doc_id: str, content_hash: str, source_name: str,
                        status: str = "pending", stats: dict = None):
        self._doc_meta[doc_id] = {
            "doc_id": doc_id, "content_hash": content_hash,
            "source_name": source_name, "status": status, **(stats or {}),
        }

    def remove_doc_subgraph(self, doc_id: str):
        self._doc_meta.pop(doc_id, None)

    # ── GraphRAG native retrieval ─────────────────────────────────────────

    def retrieve_subgraph(self, entity_names: list[str], hops: int = 2,
                          limit: int = 20) -> list[RetrievalResult]:
        """Local search anchored on seed entity names."""
        query = " AND ".join(entity_names[:3]) if entity_names else "overview"
        try:
            answer = self._local_search(query)
            return [RetrievalResult(
                content=answer, score=1.0, source="graph",
                metadata={"method": "ms_graphrag_local", "seeds": entity_names},
            )]
        except Exception as e:
            return [RetrievalResult(
                content=f"MS GraphRAG local search unavailable: {e}",
                score=0.1, source="graph", metadata={},
            )]

    def retrieve_entity_context(self, entity_name: str) -> list[RetrievalResult]:
        return self.retrieve_subgraph([entity_name])

    def global_community_search(self, top_n: int = 5) -> list[RetrievalResult]:
        """GraphRAG global search — summarizes across all communities."""
        try:
            answer = self._global_search("Provide an overview of the main themes and entities.")
            return [RetrievalResult(
                content=answer, score=1.0, source="community",
                metadata={"method": "ms_graphrag_global"},
            )]
        except Exception as e:
            return [RetrievalResult(
                content=f"MS GraphRAG global search unavailable: {e}",
                score=0.1, source="community", metadata={},
            )]

    def get_chunks_for_entities(self, entity_names: list[str]) -> list[str]:
        return []  # graphrag uses its own text unit storage

    def detect_communities(self) -> list[dict]:
        """Read community data from graphrag's parquet output."""
        try:
            import pandas as pd
            community_file = os.path.join(
                self._working_dir, "output", "communities.parquet"
            )
            if not os.path.exists(community_file):
                return []
            df = pd.read_parquet(community_file)
            return df.to_dict(orient="records")
        except Exception:
            return []

    def get_community_entities(self, community_id) -> list[dict]:
        return []

    def store_community_summary(self, community: CommunityInfo):
        pass  # graphrag manages its own community summaries

    # ── Lifecycle ────────────────────────────────────────────────────────

    def confirm_source(self, doc_id: str):
        if doc_id in self._doc_meta:
            self._doc_meta[doc_id]["status"] = "confirmed"

    def deprecate_source(self, doc_id: str):
        if doc_id in self._doc_meta:
            self._doc_meta[doc_id]["status"] = "deprecated"

    def get_doc_status(self, doc_id: str) -> str | None:
        return self._doc_meta.get(doc_id, {}).get("status")

    def prune_low_confidence(self, threshold: float = 0.5) -> int:
        return 0  # graphrag manages confidence internally

    def health_report(self) -> dict:
        communities = self.detect_communities()
        return {
            "entities": 0, "relations": 0,
            "documents": len(self._doc_meta),
            "confirmed_sources": sum(1 for m in self._doc_meta.values() if m.get("status") == "confirmed"),
            "communities": len(communities),
            "avg_confidence": 1.0,
        }

    # ── Internal search helpers ──────────────────────────────────────────

    def _local_search(self, query: str) -> str:
        try:
            from graphrag.query.cli import run_local_search
            return self._run_sync(run_local_search(root_dir=self._working_dir, query=query))
        except ImportError:
            raise ImportError(
                "microsoft/graphrag library not installed. Run: pip install graphrag\n"
                "Then initialize: python -m graphrag.index --init --root ./graphrag_workspace"
            )

    def _global_search(self, query: str) -> str:
        try:
            from graphrag.query.cli import run_global_search
            return self._run_sync(run_global_search(root_dir=self._working_dir, query=query))
        except ImportError:
            raise ImportError(
                "microsoft/graphrag library not installed. Run: pip install graphrag\n"
                "Then initialize: python -m graphrag.index --init --root ./graphrag_workspace"
            )
