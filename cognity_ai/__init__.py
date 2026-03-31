"""
raglib — Modular, provider-agnostic RAG library.

Quick start:
    from cognity_ai import RAGLibrary

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

Multimodal RAG (experimental):
    Image RAG:  CLIP / SigLIP / ImageBind / BLIP-2 — text-to-image retrieval
    Video RAG:  frame extraction, scene detection, transcript alignment
    Audio RAG:  Whisper / Google STT / AWS Transcribe — segment-level retrieval

    from cognity_ai.multimodal import ImageIngestionPipeline, ImageRetriever
    from cognity_ai.multimodal.embedders import CLIPEmbedder
    from cognity_ai.multimodal.stores import ChromaMultimodalStore

    embedder = CLIPEmbedder()
    store = ChromaMultimodalStore()
    pipeline = ImageIngestionPipeline(embedder=embedder, store=store)
    pipeline.ingest("photo.jpg")
    retriever = ImageRetriever(embedder=embedder, store=store)
    results = retriever.retrieve("a dog playing in the park")
"""

from cognity_ai.library import RAGLibrary

# Expose ABC types for users building custom plugins
from cognity_ai.loaders.base import BaseLoader
from cognity_ai.chunkers.base import BaseChunker
from cognity_ai.ocr.base import BaseOCR
from cognity_ai.page_index.base import BasePageIndex
from cognity_ai.extractors.base import BaseExtractor
from cognity_ai.embedders.base import BaseEmbedder
from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.stores.graph.base import BaseGraphStore
from cognity_ai.generators.base import BaseGenerator
from cognity_ai.retrievers.base import BaseRetriever

# Core models
from cognity_ai.models.document import Document
from cognity_ai.models.knowledge import Entity, Relation, ExtractionResult
from cognity_ai.models.retrieval import (
    RetrievalResult, SemanticChunk, CommunityInfo, PageInfo, DocumentMeta
)

# Config
from cognity_ai.config.base import LibraryConfig

# Registry for custom plugins
from cognity_ai.registry import PluginRegistry

__version__ = "2.0.0"

# ── Multimodal extension (experimental) ─────────────────────────────────────
# Expose base classes + key types so users can do:
#   from cognity_ai import BaseMultimodalEmbedder, ImageChunk, ...
from cognity_ai.multimodal.embedders.base import BaseMultimodalEmbedder
from cognity_ai.multimodal.transcribers.base import BaseTranscriber, TranscriptionResult
from cognity_ai.multimodal.stores.base import BaseMultimodalStore
from cognity_ai.multimodal.retrievers.base import BaseMultimodalRetriever
from cognity_ai.multimodal.models.media import (
    ImageChunk,
    VideoFrame,
    VideoChunk,
    AudioSegment,
    AudioChunk,
    MultimodalRetrievalResult,
)

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
    # Multimodal (experimental)
    "BaseMultimodalEmbedder", "BaseTranscriber", "TranscriptionResult",
    "BaseMultimodalStore", "BaseMultimodalRetriever",
    "ImageChunk", "VideoFrame", "VideoChunk",
    "AudioSegment", "AudioChunk", "MultimodalRetrievalResult",
    # Version
    "__version__",
]
