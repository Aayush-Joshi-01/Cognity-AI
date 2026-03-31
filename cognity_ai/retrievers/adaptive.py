"""
AdaptiveRetriever — Classifies the incoming query and routes it to the
most appropriate retriever from a provided registry.

Classification heuristics:
  - Multi-hop / relationship questions → hybrid_graph (graph-first)
  - Broad / thematic / summarization   → naive (community search via vector)
  - Factual / entity lookup            → hybrid_graph
  - Everything else                    → configurable default
"""

from cognity_ai.retrievers.base import BaseRetriever
from cognity_ai.models.retrieval import RetrievalResult


class AdaptiveRetriever(BaseRetriever):
    """
    Routes queries to the best registered retriever based on lightweight
    heuristic classification. No LLM required for routing.

    Example usage::

        adaptive = AdaptiveRetriever(
            retrievers={
                "hybrid_graph": HybridGraphRetriever(...),
                "naive": NaiveRetriever(...),
                "vector_only": VectorOnlyRetriever(...),
            },
            default="hybrid_graph",
        )
        results = adaptive.retrieve("What is the relationship between A and B?")
    """

    def __init__(
        self,
        retrievers: dict,          # {"hybrid_graph": ..., "naive": ..., ...}
        default: str = "hybrid_graph",
    ):
        self._retrievers = retrievers
        self._default = default

    # ──────────────────────────────────────────────────────────────────
    # QUERY CLASSIFICATION
    # ──────────────────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """
        Lightweight heuristic routing. Returns a retriever key.
        No LLM call — runs in microseconds.
        """
        query_lower = query.lower()

        # Multi-hop reasoning / relationship questions → graph
        if any(
            w in query_lower
            for w in [
                "relationship", "connection", "between", "how does", "why did",
                "how are", "connected to", "related to", "link between",
                "path from", "path to",
            ]
        ):
            return "hybrid_graph"

        # Broad thematic / summarization questions → community/vector search
        if any(
            w in query_lower
            for w in [
                "summarize", "summary", "overview", "themes", "main topics",
                "what are all", "list all", "general", "broadly", "in general",
                "overall", "high-level",
            ]
        ):
            return "naive"

        # Factual / entity lookup → graph-first
        if any(
            w in query_lower
            for w in [
                "who is", "who was", "what is", "what was", "when did",
                "when was", "where is", "where was", "which", "define",
                "tell me about",
            ]
        ):
            return "hybrid_graph"

        # Default
        return self._default

    # ──────────────────────────────────────────────────────────────────
    # RETRIEVAL & GENERATION
    # ──────────────────────────────────────────────────────────────────

    def _get_retriever(self, method: str) -> BaseRetriever:
        retriever = self._retrievers.get(method) or self._retrievers.get(self._default)
        if retriever is None:
            raise ValueError(
                f"No retriever found for method '{method}' and default '{self._default}'. "
                f"Available: {list(self._retrievers.keys())}"
            )
        return retriever

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        method = self._classify_query(query)
        return self._get_retriever(method).retrieve(query, top_k=top_k)

    def query(self, question: str, top_k: int = 10) -> str:
        method = self._classify_query(question)
        return self._get_retriever(method).query(question, top_k=top_k)

    def query_with_sources(self, question: str, top_k: int = 10) -> dict:
        method = self._classify_query(question)
        retriever = self._get_retriever(method)
        result = retriever.query_with_sources(question, top_k=top_k)
        result["routed_to"] = method
        return result
