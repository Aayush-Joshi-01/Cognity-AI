"""
HybridGraphRetriever — 4-channel retrieval with Reciprocal Rank Fusion.

Channels:
  1. Graph Local   — BFS subgraph expansion from seed entities (triples)
  2. Vector Semantic — cosine similarity on chunk embeddings
  3. Community Global — GraphRAG-style community summary search
  4. Graph→Vector Bridge — entity-linked chunks (graph tells vector what to fetch)

Fusion: RRF merges all channels, then confirmed-source boost applied.

Direct migration of hybrid_rag/hybrid_retriever.py HybridRetriever.
"""

from cognity_ai.retrievers.base import BaseRetriever
from cognity_ai.models.retrieval import RetrievalResult
from cognity_ai.utils.rrf import reciprocal_rank_fusion
from cognity_ai.embedders.base import BaseEmbedder
from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.stores.graph.base import BaseGraphStore
from cognity_ai.generators.base import BaseGenerator, GENERATION_PROMPT
from cognity_ai.extractors.base import BaseExtractor


class HybridGraphRetriever(BaseRetriever):
    """
    The default retriever. Fuses graph local, vector semantic, community global,
    and graph→vector bridge channels via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        nlp_extractor,           # NLPExtractor or any BaseExtractor for query parsing
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore,
        generator: BaseGenerator,
        config=None,             # LibraryConfig
    ):
        self.nlp_extractor = nlp_extractor
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.generator = generator
        self.config = config

    # ══════════════════════════════════════════════════════════════════════
    # 4-CHANNEL RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        graph_hops: int = 2,
        use_communities: bool = True,
        use_bridge: bool = True,
    ) -> list[RetrievalResult]:
        """
        Full hybrid retrieval across 4 channels + RRF fusion.
        """
        # ── Extract seed entities from query ────────────────────────────
        nlp_result = self.nlp_extractor.extract(query)
        seed_names = [e.name for e in nlp_result.entities]

        # Fallback: noun phrases / capitalised words as seeds
        if not seed_names:
            seed_names = [
                w for w in query.split()
                if len(w) > 2 and w[0].isupper()
            ]

        # ── Channel 1: Graph Local Search ───────────────────────────────
        graph_results: list[RetrievalResult] = []
        if seed_names:
            graph_results = self.graph_store.retrieve_subgraph(
                seed_names, hops=graph_hops, limit=top_k * 2
            )
            # Direct entity context for top seeds
            for name in seed_names[:3]:
                graph_results.extend(self.graph_store.retrieve_entity_context(name))

        # ── Channel 2: Vector Semantic Search ───────────────────────────
        query_emb = self.embedder.embed_query(query)
        vector_results = self.vector_store.query_chunks(query_emb, top_k=top_k)

        # ── Channel 3: Community Global Search ──────────────────────────
        community_results: list[RetrievalResult] = []
        if use_communities:
            # Graph-side community search
            global_top = 5
            if self.config is not None:
                try:
                    global_top = self.config.graphrag.global_search_top_communities
                except AttributeError:
                    pass
            community_results = self.graph_store.global_community_search(top_n=global_top)
            # Also vector-similarity on community summaries
            vec_communities = self.vector_store.query_communities(query_emb, top_k=3)
            community_results.extend(vec_communities)

        # ── Channel 4: Graph→Vector Bridge ──────────────────────────────
        bridge_results: list[RetrievalResult] = []
        if use_bridge and seed_names:
            linked_chunk_ids = self.graph_store.get_chunks_for_entities(seed_names)
            if linked_chunk_ids:
                bridge_results = self.vector_store.query_by_chunk_ids(
                    linked_chunk_ids[:top_k]
                )

        # ── RRF Fusion across all channels ──────────────────────────────
        # Weights: graph=1.2, vector=1.0, community=0.8, bridge=1.1
        merged = reciprocal_rank_fusion(
            graph_results,
            vector_results,
            community_results,
            bridge_results,
            weights=[1.2, 1.0, 0.8, 1.1],
            k=60,
        )

        # ── Confirmed-source boost ──────────────────────────────────────
        confirmed_boost = 1.5
        if self.config is not None:
            try:
                confirmed_boost = self.config.ingestion.confirmed_boost
            except AttributeError:
                pass

        boosted: list[RetrievalResult] = []
        for r in merged:
            sid = r.metadata.get("source_id") or r.metadata.get("doc_id", "")
            if sid:
                status = self.graph_store.get_doc_status(sid)
                if status == "confirmed":
                    r = r.model_copy(update={"score": r.score * confirmed_boost})
                elif status == "deprecated":
                    r = r.model_copy(update={"score": r.score * 0.5})
            boosted.append(r)

        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted[:top_k]

    # ══════════════════════════════════════════════════════════════════════
    # GENERATION
    # ══════════════════════════════════════════════════════════════════════

    def query(self, question: str, top_k: int = 10) -> str:
        results = self.retrieve(question, top_k=top_k)

        graph_ctx = "\n".join(
            r.content for r in results if r.source == "graph"
        ) or "No graph context."
        community_ctx = "\n".join(
            r.content for r in results if r.source == "community"
        ) or "No community context."
        vector_ctx = "\n---\n".join(
            r.content for r in results if r.source in ("vector", "vector_bridge")
        ) or "No document context."

        if hasattr(self.generator, "generate_rag"):
            return self.generator.generate_rag(question, graph_ctx, community_ctx, vector_ctx)
        else:
            prompt = GENERATION_PROMPT.format(
                graph_context=graph_ctx,
                community_context=community_ctx,
                vector_context=vector_ctx,
                question=question,
            )
            return self.generator.generate(question, prompt)

    def query_with_sources(self, question: str, top_k: int = 10) -> dict:
        results = self.retrieve(question, top_k=top_k)

        graph_ctx = "\n".join(
            r.content for r in results if r.source == "graph"
        ) or "None"
        community_ctx = "\n".join(
            r.content for r in results if r.source == "community"
        ) or "None"
        vector_ctx = "\n---\n".join(
            r.content for r in results if r.source in ("vector", "vector_bridge")
        ) or "None"

        if hasattr(self.generator, "generate_rag"):
            answer = self.generator.generate_rag(question, graph_ctx, community_ctx, vector_ctx)
        else:
            prompt = GENERATION_PROMPT.format(
                graph_context=graph_ctx,
                community_context=community_ctx,
                vector_context=vector_ctx,
                question=question,
            )
            answer = self.generator.generate(question, prompt)

        # Seed entities for diagnostics
        seed_entities = [e.name for e in self.nlp_extractor.extract(question).entities]

        return {
            "answer": answer,
            "sources": {
                "graph": [r.metadata for r in results if r.source == "graph"],
                "vector": [r.metadata for r in results if r.source in ("vector", "vector_bridge")],
                "community": [r.metadata for r in results if r.source == "community"],
            },
            "retrieval_scores": [
                {
                    "content": r.content[:80],
                    "score": round(r.score, 4),
                    "channel": r.source,
                }
                for r in results[:8]
            ],
            "seed_entities": seed_entities,
        }
