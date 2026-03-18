"""
raglib — Modular, provider-agnostic RAG library.

Quick start:
    from raglib import RAGLibrary

    rag = RAGLibrary(gemini_api_key="...", neo4j_password="...")
    rag.ingest("report.pdf")
    rag.ingest("slides.pptx")
    rag.build_communities()
    answer = rag.query("What are the main findings?")

Supported file formats:
    PDF, DOCX, XLSX, CSV, PPTX, TXT, MD, HTML, JSON, YAML,
    JPG, PNG, JPEG, BMP, TIFF, WEBP (via multimodal LLM OCR)

RAG methodologies:
    hybrid_graph  — 4-channel (Graph BFS + Vector + Community + Bridge) + RRF [DEFAULT]
    naive         — pure vector similarity
    vector_only   — vector + community search
    graph_only    — graph traversal only
    parent_child  — small chunk retrieval + parent context
    multi_query   — N query variants → merged results
    adaptive      — auto-routes per query type

Embedders:     gemini [default], vertex_ai, openai, azure_openai, bedrock, cohere,
               sentence_transformers, ollama
LLM providers: gemini [default], vertex_ai, openai, azure_openai, anthropic, bedrock,
               cohere, ollama
Vector stores: chroma [default], qdrant, pinecone, faiss, weaviate, milvus, pgvector,
               azure_search
Graph stores:  neo4j [default], microsoft_graphrag, memgraph, arangodb, networkx, none
OCR:           gemini_vision [default], openai_vision, anthropic_vision, azure_vision,
               bedrock_vision, tesseract
"""

from raglib.library import RAGLibrary

# Expose ABC types for users building custom plugins
from raglib.loaders.base import BaseLoader
from raglib.chunkers.base import BaseChunker
from raglib.ocr.base import BaseOCR
from raglib.page_index.base import BasePageIndex
from raglib.extractors.base import BaseExtractor
from raglib.embedders.base import BaseEmbedder
from raglib.stores.vector.base import BaseVectorStore
from raglib.stores.graph.base import BaseGraphStore
from raglib.generators.base import BaseGenerator
from raglib.retrievers.base import BaseRetriever

# Core models
from raglib.models.document import Document
from raglib.models.knowledge import Entity, Relation, ExtractionResult
from raglib.models.retrieval import (
    RetrievalResult, SemanticChunk, CommunityInfo, PageInfo, DocumentMeta
)

# Config
from raglib.config.base import LibraryConfig

# Registry for custom plugins
from raglib.registry import PluginRegistry

__version__ = "1.0.0"

__all__ = [
    # Primary API
    "RAGLibrary",
    # Abstract bases (for custom plugins)
    "BaseLoader", "BaseChunker", "BaseOCR", "BasePageIndex",
    "BaseExtractor", "BaseEmbedder", "BaseVectorStore",
    "BaseGraphStore", "BaseGenerator", "BaseRetriever",
    # Models
    "Document", "Entity", "Relation", "ExtractionResult",
    "RetrievalResult", "SemanticChunk", "CommunityInfo", "PageInfo", "DocumentMeta",
    # Config
    "LibraryConfig",
    # Plugin registry
    "PluginRegistry",
    # Version
    "__version__",
]
