"""Abstract base class for answer generators, plus the shared RAG prompt."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognity_ai.observability.collector import ObservabilityCollector
    from cognity_ai.observability.models import GenerationEvent


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


class BaseGenerator(ABC):
    # Collector is None by default — zero overhead when not configured
    _collector: "ObservabilityCollector | None" = None

    def set_collector(self, collector: "ObservabilityCollector") -> None:
        """Attach an :class:`ObservabilityCollector` to this generator."""
        self._collector = collector

    def _emit_generation(self, event: "GenerationEvent") -> None:
        """Emit a generation event if a collector is attached."""
        if self._collector is not None:
            self._collector.emit(event)

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Generate an answer given a question and a single context string."""
        ...

    def generate_with_structured_context(
        self,
        question: str,
        graph_context: str = "",
        community_context: str = "",
        vector_context: str = "",
    ) -> str:
        """Generate an answer using the structured RAG prompt with three context channels.

        Subclasses may override this for model-specific optimisations; the
        default implementation builds the prompt and delegates to generate().
        """
        prompt = self.build_rag_prompt(
            question, graph_context, community_context, vector_context
        )
        # Pass the fully-formed prompt as the context with an empty question
        # so each subclass's generate() receives a ready-to-use prompt string.
        return self.generate(question="", context=prompt)

    @staticmethod
    def build_rag_prompt(
        question: str,
        graph_ctx: str,
        community_ctx: str,
        vector_ctx: str,
    ) -> str:
        """Fill GENERATION_PROMPT with the provided context strings."""
        return GENERATION_PROMPT.format(
            graph_context=graph_ctx or "No graph context.",
            community_context=community_ctx or "No community context.",
            vector_context=vector_ctx or "No document context.",
            question=question,
        )
