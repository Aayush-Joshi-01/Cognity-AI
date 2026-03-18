"""
NaiveRetriever — Pure vector similarity search + optional generation.
No graph required. Useful as a baseline or for simple use cases.
"""

from raglib.retrievers.base import BaseRetriever
from raglib.models.retrieval import RetrievalResult
from raglib.embedders.base import BaseEmbedder
from raglib.stores.vector.base import BaseVectorStore
from raglib.generators.base import BaseGenerator


class NaiveRetriever(BaseRetriever):
    """
    Retrieves the top-k most semantically similar chunks using vector search,
    then optionally generates an answer using an LLM generator.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generator: BaseGenerator | None = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        embedding = self.embedder.embed_query(query)
        return self.vector_store.query_chunks(embedding, top_k=top_k)

    def query(self, question: str, top_k: int = 10) -> str:
        results = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(r.content for r in results)
        if self.generator:
            return self.generator.generate(question, context)
        return context
