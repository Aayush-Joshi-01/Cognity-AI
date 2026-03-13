"""
Hybrid Retriever — 4-channel retrieval with Reciprocal Rank Fusion.

Channels:
  1. Graph Local   — BFS subgraph expansion from seed entities (triples)
  2. Vector Semantic — cosine similarity on chunk embeddings
  3. Community Global — GraphRAG-style community summary search
  4. Graph→Vector Bridge — entity-linked chunks (graph tells vector what to fetch)

Fusion: RRF merges all channels, then confirmed-source boost applied.
"""

import google.generativeai as genai
from models import RetrievalResult
from nlp_processor import NLPProcessor
from gemini_extractor import GeminiExtractor
from graph_manager import GraphManager
from vector_manager import VectorManager
from config import Config


GENERATION_PROMPT = """You are a knowledgeable assistant with access to a knowledge graph and document corpus.
Use ALL provided context to give accurate, well-sourced answers. Cite specific entities and relationships.

=== Knowledge Graph Context (Entities & Relations) ===
{graph_context}

=== Community-Level Context (High-Level Themes) ===
{community_context}

=== Document Chunks (Detailed Text) ===
{vector_context}

Question: {question}

Instructions:
- Synthesize information across all context types
- Mention specific entities and their relationships
- If graph and documents conflict, note the discrepancy
- If context is insufficient, say so clearly
- Be concise but thorough

Answer:"""


class HybridRetriever:
    def __init__(self, nlp: NLPProcessor, gemini: GeminiExtractor,
                 graph: GraphManager, vector: VectorManager, config: Config):
        self.nlp = nlp
        self.gemini = gemini
        self.graph = graph
        self.vector = vector
        self.config = config

        genai.configure(api_key=config.gemini.api_key)
        self.gen_model = genai.GenerativeModel(
            config.gemini.model,
            generation_config=genai.GenerationConfig(temperature=config.gemini.temperature),
        )

    # ══════════════════════════════════════════════════════════════════════
    # 4-CHANNEL RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: int = 10,
                 graph_hops: int = 2,
                 use_communities: bool = True,
                 use_bridge: bool = True) -> list[RetrievalResult]:
        """
        Full hybrid retrieval across 4 channels + RRF fusion.
        """
        # ── Extract seed entities from query (local NLP, free) ──────────
        nlp_result = self.nlp.process(query)
        seed_names = [e.name for e in nlp_result.entities]

        # Also use noun phrases as fallback seeds
        if not seed_names:
            doc = self.nlp.nlp(query)
            seed_names = [chunk.text.title() for chunk in doc.noun_chunks
                          if len(chunk.text) > 2]

        # ── Channel 1: Graph Local Search ───────────────────────────────
        graph_results = []
        if seed_names:
            graph_results = self.graph.retrieve_subgraph(
                seed_names, hops=graph_hops, limit=top_k * 2
            )
            # Direct entity context for top seeds
            for name in seed_names[:3]:
                graph_results.extend(self.graph.retrieve_entity_context(name))

        # ── Channel 2: Vector Semantic Search ───────────────────────────
        query_emb = self.gemini.embed_query(query)
        vector_results = self.vector.query_chunks(query_emb, top_k=top_k)

        # ── Channel 3: Community Global Search ──────────────────────────
        community_results = []
        if use_communities:
            # Both graph-side and vector-side community search
            community_results = self.graph.global_community_search(
                top_n=self.config.graphrag.global_search_top_communities
            )
            # Also vector-similarity on community summaries
            vec_communities = self.vector.query_communities(query_emb, top_k=3)
            community_results.extend(vec_communities)

        # ── Channel 4: Graph→Vector Bridge ──────────────────────────────
        bridge_results = []
        if use_bridge and seed_names:
            linked_chunk_ids = self.graph.get_chunks_for_entities(seed_names)
            if linked_chunk_ids:
                bridge_results = self.vector.query_by_chunk_ids(
                    linked_chunk_ids[:top_k]
                )

        # ── RRF Fusion across all channels ──────────────────────────────
        merged = self._reciprocal_rank_fusion(
            graph_results, vector_results, community_results, bridge_results,
            k=60,
        )

        # ── Confirmed-source boost ──────────────────────────────────────
        for r in merged:
            sid = r.metadata.get("source_id") or r.metadata.get("doc_id", "")
            if sid:
                status = self.graph.get_doc_status(sid)
                if status == "confirmed":
                    r.score *= self.config.ingestion.confirmed_boost
                elif status == "deprecated":
                    r.score *= 0.5

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    # ── RRF Fusion ──────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        graph_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        community_results: list[RetrievalResult],
        bridge_results: list[RetrievalResult],
        k: int = 60,
    ) -> list[RetrievalResult]:
        """
        RRF: score(d) = Σ 1/(k + rank_i) across all lists where d appears.
        Items appearing in multiple channels get boosted naturally.
        """
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        def _add_list(results: list[RetrievalResult], prefix: str, weight: float = 1.0):
            for rank, r in enumerate(results):
                key = f"{prefix}:{r.content[:120]}"
                rrf_score = weight / (k + rank + 1)
                scores[key] = scores.get(key, 0) + rrf_score
                if key not in result_map:
                    result_map[key] = r

        _add_list(graph_results, "g", weight=1.2)     # graph gets slight boost
        _add_list(vector_results, "v", weight=1.0)
        _add_list(community_results, "c", weight=0.8)  # global context, lower weight
        _add_list(bridge_results, "b", weight=1.1)     # bridge is high-signal

        for key in result_map:
            result_map[key].score = scores[key]

        return sorted(result_map.values(), key=lambda x: x.score, reverse=True)

    # ══════════════════════════════════════════════════════════════════════
    # GENERATION
    # ══════════════════════════════════════════════════════════════════════

    def query(self, question: str, top_k: int = 10, graph_hops: int = 2) -> str:
        results = self.retrieve(question, top_k=top_k, graph_hops=graph_hops)

        graph_ctx = "\n".join(
            r.content for r in results if r.source in ("graph",)
        ) or "No graph context."
        community_ctx = "\n".join(
            r.content for r in results if r.source == "community"
        ) or "No community context."
        vector_ctx = "\n---\n".join(
            r.content for r in results if r.source in ("vector", "vector_bridge")
        ) or "No document context."

        prompt = GENERATION_PROMPT.format(
            graph_context=graph_ctx, community_context=community_ctx,
            vector_context=vector_ctx, question=question,
        )
        resp = self.gen_model.generate_content(prompt)
        return resp.text

    def query_with_sources(self, question: str, top_k: int = 10) -> dict:
        results = self.retrieve(question, top_k=top_k)

        graph_ctx = "\n".join(r.content for r in results if r.source == "graph") or "None"
        community_ctx = "\n".join(r.content for r in results if r.source == "community") or "None"
        vector_ctx = "\n---\n".join(
            r.content for r in results if r.source in ("vector", "vector_bridge")
        ) or "None"

        prompt = GENERATION_PROMPT.format(
            graph_context=graph_ctx, community_context=community_ctx,
            vector_context=vector_ctx, question=question,
        )
        resp = self.gen_model.generate_content(prompt)

        return {
            "answer": resp.text,
            "sources": {
                "graph": [r.metadata for r in results if r.source == "graph"],
                "vector": [r.metadata for r in results if r.source in ("vector", "vector_bridge")],
                "community": [r.metadata for r in results if r.source == "community"],
            },
            "retrieval_scores": [
                {"content": r.content[:80], "score": round(r.score, 4), "channel": r.source}
                for r in results[:8]
            ],
            "seed_entities": [e.name for e in self.nlp.process(question).entities],
        }
