"""
ParentChildRetriever — Retrieves small child chunks then returns their parent
chunk context for richer surrounding text.

Requires a vector store that has both parent and child chunks stored with
appropriate metadata ("is_parent", "parent_chunk_id").
"""

from raglib.retrievers.base import BaseRetriever
from raglib.models.retrieval import RetrievalResult
from raglib.embedders.base import BaseEmbedder
from raglib.stores.vector.base import BaseVectorStore
from raglib.generators.base import BaseGenerator


class ParentChildRetriever(BaseRetriever):
    """
    Two-stage retrieval: first finds precise child chunks via vector search,
    then returns the larger parent chunks that contain those children for
    broader context.
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
        emb = self.embedder.embed_query(query)

        # Retrieve child chunks (small, precise matches)
        child_results = self.vector_store.query_chunks(
            emb, top_k=top_k, filters={"is_parent": False}
        )

        if not child_results:
            # Fall back to regular retrieval if no child chunks available
            return self.vector_store.query_chunks(emb, top_k=top_k)

        # Collect unique parent chunk IDs from matched children
        parent_ids: list[str] = []
        for r in child_results:
            parent_id = r.metadata.get("parent_chunk_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        if parent_ids:
            # Return parent chunks (larger context windows)
            parent_results = self.vector_store.query_by_chunk_ids(parent_ids)
            return parent_results[:top_k]

        return child_results[:top_k]

    def query(self, question: str, top_k: int = 10) -> str:
        results = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(r.content for r in results)
        if self.generator:
            return self.generator.generate(question, context)
        return context
