import os
from dataclasses import dataclass, field


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "neo4j+s://xxxxx.databases.neo4j.io")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class GeminiConfig:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.1
    extraction_temperature: float = 0.0
    batch_embed_limit: int = 100
    rpm_limit: int = 15


@dataclass
class ChromaConfig:
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    collection_name: str = "hybrid_rag_chunks"
    community_collection: str = "community_summaries"


@dataclass
class NLPConfig:
    spacy_model: str = "en_core_web_trf"
    fallback_model: str = "en_core_web_sm"
    min_entity_freq: int = 1
    dependency_relations: list[str] = field(default_factory=lambda: [
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
    max_gemini_chunks_per_doc: int = 50
    cache_embeddings: bool = True


@dataclass
class Config:
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
