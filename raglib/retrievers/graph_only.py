"""
GraphOnlyRetriever — Pure graph traversal retrieval, no vector search.
Useful when only a knowledge graph is available or to isolate graph-based recall.
"""

from raglib.retrievers.base import BaseRetriever
from raglib.models.retrieval import RetrievalResult
from raglib.stores.graph.base import BaseGraphStore
from raglib.generators.base import BaseGenerator
from raglib.extractors.base import BaseExtractor


class GraphOnlyRetriever(BaseRetriever):
    """
    Retrieves context entirely from graph traversal.
    Uses NLP extraction to identify seed entities in the query, then performs
    BFS subgraph expansion and entity context lookup.
    """

    def __init__(
        self,
        nlp_extractor: BaseExtractor,
        graph_store: BaseGraphStore,
        generator: BaseGenerator | None = None,
    ):
        self.nlp_extractor = nlp_extractor
        self.graph_store = graph_store
        self.generator = generator

    def retrieve(self, query: str, top_k: int = 10, hops: int = 2) -> list[RetrievalResult]:
        result = self.nlp_extractor.extract(query)
        seed_names = [e.name for e in result.entities]

        # Fallback: extract capitalised words as seeds
        if not seed_names:
            seed_names = [w for w in query.split() if len(w) > 3 and w[0].isupper()]

        graph_results: list[RetrievalResult] = []
        if seed_names:
            graph_results = self.graph_store.retrieve_subgraph(
                seed_names, hops=hops, limit=top_k * 2
            )
            for name in seed_names[:3]:
                graph_results.extend(self.graph_store.retrieve_entity_context(name))

        # Deduplicate by content
        seen: set[str] = set()
        unique: list[RetrievalResult] = []
        for r in graph_results:
            if r.content not in seen:
                seen.add(r.content)
                unique.append(r)

        return unique[:top_k]

    def query(self, question: str, top_k: int = 10) -> str:
        results = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(r.content for r in results)
        if self.generator:
            return self.generator.generate(question, context)
        return context
