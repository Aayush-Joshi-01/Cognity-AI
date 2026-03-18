"""Abstract base class for retrieval strategies."""
from abc import ABC, abstractmethod
from raglib.models.retrieval import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...

    def query(self, question: str, top_k: int = 10) -> str:
        """Retrieve and return concatenated content. Subclasses with generators should override."""
        results = self.retrieve(question, top_k=top_k)
        return "\n\n".join(r.content for r in results[:top_k])

    def query_with_sources(self, question: str, top_k: int = 10) -> dict:
        results = self.retrieve(question, top_k=top_k)
        return {
            "answer": self.query(question, top_k=top_k),
            "sources": {
                "graph": [r.metadata for r in results if r.source == "graph"],
                "vector": [r.metadata for r in results if r.source in ("vector", "vector_bridge")],
                "community": [r.metadata for r in results if r.source == "community"],
            },
            "retrieval_scores": [
                {"content": r.content[:80], "score": round(r.score, 4), "channel": r.source}
                for r in results[:8]
            ],
        }
