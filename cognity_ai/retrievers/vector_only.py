"""
VectorOnlyRetriever — Vector similarity search with optional community search.

Similar to NaiveRetriever but explicitly "vector only" and includes community
summary search (community summaries are stored in the vector collection too).
"""

from cognity_ai.retrievers.base import BaseRetriever
from cognity_ai.models.retrieval import RetrievalResult
from cognity_ai.embedders.base import BaseEmbedder
from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.generators.base import BaseGenerator
from cognity_ai.utils.rrf import reciprocal_rank_fusion


class VectorOnlyRetriever(BaseRetriever):
    """
    Retrieves using only vector similarity. Optionally includes community
    summaries from the vector store, fused via RRF.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generator: BaseGenerator | None = None,
        include_communities: bool = True,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.include_communities = include_communities

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        emb = self.embedder.embed_query(query)
        results = self.vector_store.query_chunks(emb, top_k=top_k)

        if self.include_communities:
            community_results = self.vector_store.query_communities(emb, top_k=3)
            # Use RRF to merge chunks and community results
            results = reciprocal_rank_fusion(
                results, community_results, weights=[1.0, 0.8]
            )

        return results[:top_k]

    def query(self, question: str, top_k: int = 10) -> str:
        results = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(r.content for r in results)
        if self.generator:
            return self.generator.generate(question, context)
        return context
