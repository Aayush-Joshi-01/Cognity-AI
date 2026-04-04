"""Top-level LibraryConfig that holds all component selections and provider configs."""
from __future__ import annotations
from dataclasses import dataclass, field
from cognity_ai.config.providers import (
    Neo4jConfig, GeminiConfig, ChromaConfig, NLPConfig,
    GraphRAGConfig, IngestionConfig, OpenAIConfig, AnthropicConfig,
    AzureOpenAIConfig, BedrockConfig, VertexAIConfig, QdrantConfig,
    PineconeConfig, MilvusConfig, WeaviateConfig, PgVectorConfig,
    AzureSearchConfig, OllamaConfig, CohereConfig, ObservabilityConfig,
)


@dataclass
class LibraryConfig:
    # ── Component selections ──────────────────────────────────────────────
    rag_method: str = "hybrid_graph"
    # hybrid_graph | naive | parent_child | multi_query | graph_only | vector_only
    # microsoft_graphrag | adaptive

    chunker: str = "sentence"
    # sentence | fixed | semantic | recursive | parent_child | hybrid

    embedder: str = "gemini"
    # gemini | vertex_ai | openai | azure_openai | bedrock | cohere
    # sentence_transformers | ollama

    vector_store: str = "chroma"
    # chroma | qdrant | pinecone | faiss | weaviate | milvus | pgvector | azure_search

    graph_store: str = "neo4j"
    # neo4j | microsoft_graphrag | memgraph | arangodb | networkx | none

    llm: str = "gemini"
    # gemini | vertex_ai | openai | azure_openai | anthropic | bedrock | cohere | ollama

    extraction: str = "hybrid"
    # hybrid | nlp_only | llm_only

    ocr: str = "gemini_vision"
    # gemini_vision | openai_vision | anthropic_vision | azure_vision | bedrock_vision | tesseract

    page_index: str = "hybrid"
    # hybrid | regex | structural

    # ── Provider configs ─────────────────────────────────────────────────
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    vertex_ai: VertexAIConfig = field(default_factory=VertexAIConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    cohere: CohereConfig = field(default_factory=CohereConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    weaviate: WeaviateConfig = field(default_factory=WeaviateConfig)
    pgvector: PgVectorConfig = field(default_factory=PgVectorConfig)
    azure_search: AzureSearchConfig = field(default_factory=AzureSearchConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


@dataclass
class MinimalLibraryConfig(LibraryConfig):
    """Bare-minimum config using only local, dependency-light components.

    Safe to instantiate without any API keys or running external services.
    Components that require network access (embedders, generators) still need
    credentials — those tests are automatically skipped when credentials are absent.

    Defaults:
      - vector_store: faiss    (in-memory, no server needed)
      - graph_store:  networkx (in-memory, no server needed)
      - chunker:      fixed    (no spaCy dependency)
      - extraction:   llm_only (no spaCy dependency)
      - rag_method:   vector_only (no graph store required)
      - ocr:          tesseract   (local; skipped in tests if pytesseract missing)
      - page_index:   regex       (pure Python)
    """
    rag_method: str = "vector_only"
    chunker: str = "fixed"
    embedder: str = "gemini"          # override per-test when key unavailable
    vector_store: str = "faiss"
    graph_store: str = "networkx"
    llm: str = "gemini"               # override per-test when key unavailable
    extraction: str = "llm_only"
    ocr: str = "tesseract"
    page_index: str = "regex"
