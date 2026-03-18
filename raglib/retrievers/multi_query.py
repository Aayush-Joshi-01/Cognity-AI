"""
MultiQueryRetriever — Generates N query variants from the original query,
retrieves for each variant, deduplicates, and reranks via RRF.

Improves recall by overcoming vocabulary mismatch between the user's phrasing
and the indexed document text.
"""

from raglib.retrievers.base import BaseRetriever
from raglib.models.retrieval import RetrievalResult
from raglib.embedders.base import BaseEmbedder
from raglib.stores.vector.base import BaseVectorStore
from raglib.stores.graph.base import BaseGraphStore
from raglib.generators.base import BaseGenerator
from raglib.utils.rrf import reciprocal_rank_fusion


class MultiQueryRetriever(BaseRetriever):
    """
    Uses an LLM to generate N alternative phrasings of the query, retrieves
    results for each phrasing, then merges with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generator: BaseGenerator,
        graph_store: BaseGraphStore | None = None,
        n_queries: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.graph_store = graph_store
        self.n_queries = n_queries

    def _generate_queries(self, query: str) -> list[str]:
        """Use LLM to generate N alternative phrasings of the query."""
        prompt = (
            f"Generate {self.n_queries} different ways to ask the following question. "
            f"Return only the questions, one per line, no numbering.\n\n"
            f"Original: {query}\n\n"
            f"Alternative questions:"
        )
        response = self.generator.generate(query, prompt)
        variants = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Always include the original query first
        return [query] + variants[: self.n_queries - 1]

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        queries = self._generate_queries(query)

        all_results: list[list[RetrievalResult]] = []
        for q in queries:
            emb = self.embedder.embed_query(q)
            results = self.vector_store.query_chunks(emb, top_k=top_k)
            all_results.append(results)

        # Merge all result lists with equal weights via RRF
        if not all_results:
            return []
        return reciprocal_rank_fusion(*all_results)[:top_k]

    def query(self, question: str, top_k: int = 10) -> str:
        results = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(r.content for r in results)
        return self.generator.generate(question, context)
