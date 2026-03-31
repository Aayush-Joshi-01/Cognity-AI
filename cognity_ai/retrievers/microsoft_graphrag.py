"""
MicrosoftGraphRAGRetriever — Wraps the microsoft/graphrag library.

Requires:
  pip install graphrag

Set up a graphrag workspace first:
  graphrag init --root ./graphrag_workspace
  graphrag index --root ./graphrag_workspace

This is a best-effort wrapper; the graphrag API may vary by version.
"""

from cognity_ai.retrievers.base import BaseRetriever
from cognity_ai.models.retrieval import RetrievalResult
from cognity_ai.embedders.base import BaseEmbedder
from cognity_ai.generators.base import BaseGenerator


class MicrosoftGraphRAGRetriever(BaseRetriever):
    """
    Uses microsoft/graphrag library for local or global search.

    - "local"  search: entity-centric, uses the graph neighbourhood
    - "global" search: community-centric, uses high-level summaries
    """

    def __init__(
        self,
        working_dir: str = "./graphrag_workspace",
        search_type: str = "local",   # "local" | "global"
        embedder: BaseEmbedder | None = None,
        generator: BaseGenerator | None = None,
    ):
        self._working_dir = working_dir
        self._search_type = search_type
        self._embedder = embedder
        self._generator = generator

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """
        Attempts to call graphrag search and wraps the text result as a
        single RetrievalResult. The graphrag library does not natively return
        ranked chunks, so the full answer text is returned as one result.
        """
        try:
            import asyncio
            answer = asyncio.run(self._run_search(query))
            return [
                RetrievalResult(
                    content=answer,
                    score=1.0,
                    source="graphrag",
                    metadata={"search_type": self._search_type, "working_dir": self._working_dir},
                )
            ]
        except ImportError:
            raise ImportError(
                "microsoft/graphrag library not installed. Run: pip install graphrag"
            )

    def query(self, question: str, top_k: int = 10) -> str:
        try:
            import asyncio
            return asyncio.run(self._run_search(question))
        except ImportError:
            raise ImportError(
                "microsoft/graphrag library not installed. Run: pip install graphrag"
            )

    async def _run_search(self, question: str) -> str:
        """Delegate to graphrag's local or global search CLI helpers."""
        try:
            # graphrag >= 0.3 exposes these async helpers
            from graphrag.query.cli import run_local_search, run_global_search  # type: ignore[import]

            if self._search_type == "global":
                result = await run_global_search(root_dir=self._working_dir, query=question)
            else:
                result = await run_local_search(root_dir=self._working_dir, query=question)

            # result may be a string or an object with a .response attribute
            if isinstance(result, str):
                return result
            if hasattr(result, "response"):
                return result.response
            return str(result)

        except ImportError:
            raise ImportError(
                "microsoft/graphrag library not installed. Run: pip install graphrag"
            )
        except Exception as exc:
            return f"GraphRAG search error: {exc}"
