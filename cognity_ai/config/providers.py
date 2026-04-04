"""Provider-specific configuration dataclasses."""
import os
from dataclasses import dataclass, field


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class GeminiConfig:
    # API key — checks GOOGLE_API_KEY first (new SDK default), then GEMINI_API_KEY for compat.
    # Leave empty to rely entirely on env-var auto-loading inside the SDK.
    api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    )
    model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.1
    extraction_temperature: float = 0.0
    batch_embed_limit: int = 100
    rpm_limit: int = 15
    # Vertex AI / project-based access
    project_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    location: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
    use_vertexai: bool = False
    # HTTP timeout in seconds for API calls
    timeout: int = 120


@dataclass
class VertexAIConfig:
    project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    model: str = "gemini-1.5-pro"
    embedding_model: str = "text-embedding-005"
    temperature: float = 0.1


@dataclass
class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.1
    embedding_dimensions: int = 1536


@dataclass
class AzureOpenAIConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_OPENAI_KEY", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
    temperature: float = 0.1


@dataclass
class AnthropicConfig:
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class BedrockConfig:
    region: str = os.getenv("AWS_REGION", "us-east-1")
    access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    temperature: float = 0.1


@dataclass
class CohereConfig:
    api_key: str = os.getenv("COHERE_API_KEY", "")
    model: str = "command-r-plus"
    embedding_model: str = "embed-english-v3.0"
    temperature: float = 0.1


@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = "llama3"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.1


@dataclass
class ChromaConfig:
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    collection_name: str = "raglib_chunks"
    community_collection: str = "raglib_communities"


@dataclass
class QdrantConfig:
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = "raglib_chunks"
    community_collection: str = "raglib_communities"
    vector_size: int = 768  # Set based on embedder


@dataclass
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX", "raglib")
    namespace: str = "chunks"
    community_namespace: str = "communities"
    dimension: int = 1536


@dataclass
class MilvusConfig:
    uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    token: str = os.getenv("MILVUS_TOKEN", "")
    collection_name: str = "raglib_chunks"
    community_collection: str = "raglib_communities"
    dimension: int = 768


@dataclass
class WeaviateConfig:
    url: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key: str = os.getenv("WEAVIATE_API_KEY", "")
    class_name: str = "RaglibChunk"
    community_class: str = "RaglibCommunity"


@dataclass
class PgVectorConfig:
    dsn: str = os.getenv("PGVECTOR_DSN", "postgresql://localhost/raglib")
    table_name: str = "raglib_chunks"
    community_table: str = "raglib_communities"
    dimension: int = 768


@dataclass
class AzureSearchConfig:
    endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_SEARCH_KEY", "")
    index_name: str = os.getenv("AZURE_SEARCH_INDEX", "raglib-chunks")
    community_index: str = "raglib-communities"
    dimension: int = 1536


@dataclass
class NLPConfig:
    spacy_model: str = "en_core_web_trf"
    fallback_model: str = "en_core_web_sm"
    min_entity_freq: int = 1
    dependency_relations: list = field(default_factory=lambda: [
        "nsubj", "dobj", "pobj", "attr", "prep", "agent",
        "nsubjpass", "appos", "compound", "amod",
    ])
    semantic_chunk_sentences: int = 5
    semantic_chunk_overlap: int = 1


@dataclass
class GraphRAGConfig:
    leiden_resolution: float = 1.0
    max_community_levels: int = 3
    community_summary_max_tokens: int = 300
    local_search_top_k: int = 10
    global_search_top_communities: int = 5


@dataclass
class IngestionConfig:
    hash_store_path: str = "./doc_hashes.json"
    page_index_path: str = "./page_index.json"
    confidence_threshold: float = 0.5
    confirmed_boost: float = 1.5
    use_local_nlp_first: bool = True
    gemini_extraction_mode: str = "augment"  # "augment" | "full"
    max_llm_chunks_per_doc: int = 50
    cache_embeddings: bool = True


@dataclass
class ObservabilityConfig:
    """Configuration for the AI observability layer.

    Attributes:
        enabled: Master switch — set to False to make all emit calls no-ops.
        observer: Built-in observer name: ``"noop"`` or ``"logging"``.
            Pass a custom :class:`BaseObserver` instance directly to
            :class:`RAGLibrary` for third-party integrations.
        log_level: Logging level used by ``LoggingObserver`` (default "INFO").
        max_event_buffer: Maximum number of recent events kept in memory.
    """
    enabled: bool = True
    observer: str = "noop"       # "noop" | "logging"
    log_level: str = "INFO"
    max_event_buffer: int = 1000
