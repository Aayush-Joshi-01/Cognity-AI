# cognity-ai API Reference

Complete public API documentation for the `cognity-ai` package.

---

## Table of Contents

- [1. RAGLibrary — Primary API](#1-raglibrary--primary-api)
  - [Constructor](#constructor)
  - [Ingestion Methods](#ingestion-methods)
  - [Query Methods](#query-methods)
  - [Knowledge Lifecycle Methods](#knowledge-lifecycle-methods)
  - [Plugin Registration Methods](#plugin-registration-methods)
  - [Properties](#properties)
- [2. Data Models](#2-data-models)
  - [Document](#document)
  - [ImageRef](#imageref)
  - [Entity](#entity)
  - [Relation](#relation)
  - [ExtractionResult](#extractionresult)
  - [PageInfo](#pageinfo)
  - [SemanticChunk](#semanticchunk)
  - [CommunityInfo](#communityinfo)
  - [RetrievalResult](#retrievalresult)
  - [SourceStatus](#sourcestatus)
- [3. Configuration](#3-configuration)
  - [LibraryConfig — Top-Level Fields](#libraryconfig--top-level-fields)
  - [Provider Configs](#provider-configs)
- [4. Abstract Base Classes](#4-abstract-base-classes)
  - [BaseLoader](#baseloader)
  - [BaseOCR](#baseocr)
  - [BaseChunker](#basechunker)
  - [BaseEmbedder](#baseembedder)
  - [BaseVectorStore](#basevectorstore)
  - [BaseGraphStore](#basegraphstore)
  - [BaseGenerator](#basegenerator)
  - [BaseRetriever](#baseretriever)
- [5. PluginRegistry](#5-pluginregistry)
- [6. Utility Modules](#6-utility-modules)
  - [HashStore](#hashstore)
  - [reciprocal_rank_fusion](#reciprocal_rank_fusion)
  - [Token Counter](#token-counter)
- [7. PDF Utilities](#7-pdf-utilities)
- [8. Backward Compatibility](#8-backward-compatibility)

---

## 1. RAGLibrary — Primary API

**Module:** `cognity-ai/library.py`

`RAGLibrary` is the single entry point for all cognity-ai operations. It wires together loaders, chunkers, embedders, stores, extractors, and retrievers based on the provided configuration. All component selection is done by name string, making it easy to swap providers without touching application code.

### Constructor

```python
RAGLibrary(
    rag_method="hybrid_graph",    # hybrid_graph|naive|vector_only|graph_only|parent_child|multi_query|microsoft_graphrag|adaptive
    chunker="sentence",           # sentence|fixed|recursive|semantic|parent_child|hybrid
    embedder="gemini",            # gemini|vertex_ai|openai|azure_openai|bedrock|cohere|sentence_transformers|ollama
    vector_store="chroma",        # chroma|qdrant|pinecone|faiss|weaviate|milvus|pgvector|azure_search
    graph_store="neo4j",          # neo4j|microsoft_graphrag|memgraph|arangodb|networkx|none
    llm="gemini",                 # gemini|vertex_ai|openai|azure_openai|anthropic|bedrock|cohere|ollama
    extraction="hybrid",          # hybrid|nlp_only|llm_only
    ocr="gemini_vision",          # gemini_vision|openai_vision|anthropic_vision|azure_vision|bedrock_vision|tesseract
    page_index="hybrid",          # hybrid|regex|structural
    gemini_api_key="",
    openai_api_key="",
    anthropic_api_key="",
    azure_openai_endpoint="",
    azure_openai_key="",
    azure_openai_deployment="gpt-4o",
    azure_openai_api_version="2024-02-01",
    aws_region="us-east-1",
    aws_access_key_id="",
    aws_secret_access_key="",
    cohere_api_key="",
    ollama_base_url="http://localhost:11434",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="",
    neo4j_database="neo4j",
    chroma_persist_dir="./chroma_store",
    config=None,                  # Full LibraryConfig override
)
```

**Parameter notes:**

- All string selector parameters (`rag_method`, `chunker`, etc.) must be one of the listed valid values. An unknown value raises `ValueError` at construction time.
- API key parameters are convenience shortcuts that populate the corresponding provider config fields. Passing `config=` with a fully constructed `LibraryConfig` overrides all other keyword arguments.
- When `graph_store="none"`, graph-dependent features (community detection, graph retrieval, knowledge lifecycle) are disabled. `RAGLibrary.updater` will be `None`.

**Minimal usage:**

```python
from cognity_ai import RAGLibrary

lib = RAGLibrary(
    gemini_api_key="AIza...",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="secret",
)
lib.ingest("report.pdf")
answer = lib.query("What are the key findings?")
```

**Switching to OpenAI + Qdrant:**

```python
lib = RAGLibrary(
    rag_method="vector_only",
    embedder="openai",
    llm="openai",
    vector_store="qdrant",
    graph_store="none",
    openai_api_key="sk-...",
)
```

---

### Ingestion Methods

#### `ingest(source, doc_id=None, status="pending", **meta) → dict`

Ingest a single file. The file format is auto-detected from the extension. If `doc_id` is not provided, a stable ID is derived from the file path. Additional keyword arguments are stored as document metadata.

Returns a status dict whose shape depends on the outcome:

```python
# Newly ingested or re-ingested after change:
{"doc_id": "report", "status": "ingested", "pages": 5, "chunks": 23, "entities": 47, "relations": 31}

# File content is identical to the last ingest (SHA-256 hash match):
{"doc_id": "report", "status": "skipped", "reason": "unchanged"}

# Multi-page or multi-sheet source (PDF with many pages, Excel with many sheets):
{"doc_id": "data", "status": "ingested", "parts": 3, "total_entities": 85, "total_chunks": 60}
```

**Supported extensions** (built-in loaders): `.pdf`, `.docx`, `.txt`, `.html`, `.htm`, `.csv`, `.xlsx`, `.xls`, `.json`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`

Additional formats can be added via `register_loader()`.

---

#### `ingest_dir(directory, glob="**/*", status="pending", recursive=True, **meta) → list[dict]`

Batch-ingest all matching files under `directory`. Returns a list of per-file status dicts in the same format as `ingest()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | `str` | — | Root directory to scan |
| `glob` | `str` | `"**/*"` | Glob pattern relative to `directory` |
| `status` | `str` | `"pending"` | Initial source status for all documents |
| `recursive` | `bool` | `True` | Whether to recurse into subdirectories |
| `**meta` | — | — | Metadata applied to all documents |

```python
results = lib.ingest_dir("./docs", glob="**/*.pdf", department="legal")
```

---

#### `ingest_text(text, doc_id, source_name="", status="pending") → dict`

Ingest raw text without a file on disk. Provided for backward compatibility with workflows that produce text programmatically.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Full document text |
| `doc_id` | `str` | Unique identifier for this document |
| `source_name` | `str` | Human-readable label shown in source attribution |
| `status` | `str` | `"pending"` \| `"confirmed"` \| `"deprecated"` |

---

#### `ingest_batch(documents) → list[dict]`

Ingest a list of documents in one call. Each element must be a dict with at least `doc_id` and `text`. Optional keys: `source_name`, `status`.

```python
lib.ingest_batch([
    {"doc_id": "a", "text": "...", "source_name": "Doc A"},
    {"doc_id": "b", "text": "...", "status": "confirmed"},
])
```

---

#### `build_communities() → list[CommunityInfo]`

Run Leiden/Louvain community detection over the graph, generate LLM summaries for each detected cluster, embed the summaries, and persist them to both the graph store and vector store. Call this after bulk ingestion when the graph is sufficiently populated.

Returns the list of `CommunityInfo` objects that were created.

---

#### `remove_document(doc_id)`

Remove a document and all associated data: graph nodes and edges, vector embeddings, page index entries, and the hash store entry. Subsequent calls to `ingest()` with the same `doc_id` will treat the document as new.

---

#### `sync(current_doc_ids) → list[str]`

Garbage-collect documents that are no longer in the provided set. Any document present in the internal hash store but absent from `current_doc_ids` is fully removed (equivalent to calling `remove_document()` for each).

Returns the list of `doc_id` values that were removed.

```python
# Keep only what is currently on disk
on_disk = {p.stem for p in Path("./docs").glob("*.pdf")}
removed = lib.sync(on_disk)
```

---

### Query Methods

#### `query(question, top_k=10, method=None) → str`

Full retrieval-augmented generation. Runs the retriever then passes the context to the configured LLM generator. Returns the generated answer as a plain string.

`method` overrides the library's default `rag_method` for this single call, useful for A/B comparisons without re-constructing the library.

---

#### `query_with_sources(question, top_k=10, method=None) → dict`

Full RAG with complete source attribution. Returns:

```python
{
    "answer": str,                   # LLM-generated answer
    "sources": {
        "graph": [
            {"content": str, "score": float, "metadata": dict},
            ...
        ],
        "vector": [...],
        "community": [...],
    },
    "retrieval_scores": [            # All retrieved results ranked post-fusion
        {"content": str, "score": float, "channel": str},
        ...
    ],
    "seed_entities": [str, ...],     # Named entities extracted from the question
}
```

---

#### `retrieve(query, top_k=10, method=None) → list[RetrievalResult]`

Retrieval only — no generation. Returns the fused and ranked list of `RetrievalResult` objects. Useful for building custom generation layers or evaluation pipelines.

---

### Knowledge Lifecycle Methods

These methods manage the confidence and status of ingested knowledge. They require a graph store (i.e., `graph_store` must not be `"none"`).

#### `confirm(doc_id)`

Mark a document source as confirmed. Boosts all triple confidences associated with this document to `1.0` and sets the document status to `"confirmed"`.

---

#### `deprecate(doc_id)`

Mark a document source as deprecated. Halves all triple confidences associated with this document and sets the document status to `"deprecated"`. Deprecated triples remain queryable but are penalized during retrieval scoring.

---

#### `bulk_confirm(doc_ids)`

Batch version of `confirm()`. Accepts a list of `doc_id` strings.

---

#### `bulk_deprecate(doc_ids)`

Batch version of `deprecate()`. Accepts a list of `doc_id` strings.

---

#### `detect_conflicts(entity_name) → list[dict]`

Find contradictory relations for a given entity across different source documents. A conflict is detected when two or more sources assert different target values for the same `(subject, relation_type)` pair.

```python
conflicts = lib.detect_conflicts("Anthropic")
# [
#   {
#     "entity": "Anthropic",
#     "relation": "RAISED",
#     "versions": [
#       {"src": "Anthropic", "rel": "RAISED", "tgt": "$7B",  "source_id": "doc_old", "confidence": 0.5},
#       {"src": "Anthropic", "rel": "RAISED", "tgt": "$10B", "source_id": "doc_new", "confidence": 1.0},
#     ]
#   }
# ]
```

---

#### `prune(threshold=0.5) → int`

Delete all relations with `confidence < threshold` from the graph store. Returns the number of relations removed.

---

#### `health_report() → dict`

Return a snapshot of knowledge base health:

```python
{
    "entities": int,
    "relations": int,
    "documents": int,
    "communities": int,
    "avg_confidence": float,
}
```

---

#### `source_stats() → list[dict]`

Return per-document statistics including doc_id, source name, status, and triple/chunk/entity/relation counts.

---

### Plugin Registration Methods

#### `register_loader(ext, loader_class)`

Register a custom file loader for a given extension. `ext` must include the leading dot (e.g., `".myext"`). `loader_class` must be a subclass of `BaseLoader`.

```python
lib.register_loader(".myext", MyCustomLoader)
lib.ingest("file.myext")
```

---

#### `register_chunker(name, chunker_class)`

Register a custom chunker. `chunker_class` must be a subclass of `BaseChunker`.

---

#### `register_embedder(name, embedder_class)`

Register a custom embedder. `embedder_class` must be a subclass of `BaseEmbedder`.

---

#### `register_generator(name, generator_class)`

Register a custom LLM generator. `generator_class` must be a subclass of `BaseGenerator`.

---

#### `register_retriever(name, retriever_class)`

Register a custom retriever strategy. `retriever_class` must be a subclass of `BaseRetriever`.

---

#### `available_plugins() → dict`

Return a summary of all registered plugins across all types:

```python
{
    "loaders":    [".pdf", ".docx", ".myext", ...],
    "chunkers":   ["sentence", "fixed", "recursive", ...],
    "embedders":  ["gemini", "openai", ...],
    "generators": ["gemini", "anthropic", ...],
    "retrievers": ["hybrid_graph", "vector_only", ...],
    "ocr":        ["gemini_vision", "tesseract", ...],
}
```

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.pipeline` | `IngestionPipeline` | Direct access to the ingestion pipeline |
| `.retriever` | `BaseRetriever` | Direct access to the active retriever instance |
| `.updater` | `KnowledgeUpdater \| None` | Direct access to the lifecycle manager; `None` when `graph_store="none"` |
| `.config` | `LibraryConfig` | The resolved configuration object |

---

## 2. Data Models

**Module:** `cognity-ai/models/`

Models are dataclasses (not Pydantic) unless otherwise noted. Legacy code that used Pydantic v2 models from `models.py` will still work; those classes have been migrated here.

---

### Document

**Module:** `cognity-ai/models/document.py`

```python
@dataclass
class Document:
    doc_id: str
    text: str
    source_path: str = ""
    source_name: str = ""
    loader: str = ""                          # Name of the loader that produced this document
    metadata: dict = field(default_factory=dict)
    page_map: list[PageInfo] = field(default_factory=list)
    image_refs: list[ImageRef] = field(default_factory=list)
    file_extension: str = ""
    file_size_bytes: int = 0
    page_count: int = 0
    char_count: int = 0                       # Auto-computed from len(text)
```

`char_count` is set automatically by `__post_init__` if not explicitly provided.

---

### ImageRef

**Module:** `cognity-ai/models/document.py`

Represents an image embedded within a document, with its position in the extracted text stream.

```python
@dataclass
class ImageRef:
    image_id: str
    char_offset: int          # Position in text where this image logically appears
    image_bytes: bytes = b""
    mime_type: str = "image/png"
    page_num: int = 0
    caption: str = ""
    ocr_text: str = ""        # Populated after OCR processing
```

---

### Entity

**Module:** `cognity-ai/models/knowledge.py`

```python
@dataclass
class Entity:
    name: str                          # Canonical name (title-cased)
    entity_type: str                   # "Person", "Organization", "Concept", etc.
    description: str = ""
    confidence: float = 1.0            # 0.0–1.0
    mentions: int = 1                  # Cumulative frequency across chunks
    source_id: str = ""                # Origin document ID
    extraction_method: str = "nlp"     # "nlp" | "llm" | "merged" | "nlp_np" | "nlp_svo"
    properties: dict = field(default_factory=dict)
```

---

### Relation

**Module:** `cognity-ai/models/knowledge.py`

```python
@dataclass
class Relation:
    source_entity: str                 # Source entity name
    relation_type: str                 # UPPER_SNAKE_CASE label
    target_entity: str                 # Target entity name
    description: str = ""
    confidence: float = 1.0
    weight: float = 1.0                # Accumulates on deduplication
    source_id: str = ""
    extraction_method: str = "nlp"
```

---

### ExtractionResult

**Module:** `cognity-ai/models/knowledge.py`

```python
@dataclass
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
```

---

### PageInfo

**Module:** `cognity-ai/models/document.py`

```python
@dataclass
class PageInfo:
    page_num: int
    section: str = ""
    start_char: int = 0
    end_char: int = 0
    heading: str = ""
```

---

### SemanticChunk

**Module:** `cognity-ai/models/retrieval.py`

```python
@dataclass
class SemanticChunk:
    chunk_id: str                      # "{doc_id}__chunk_{index}"
    doc_id: str
    text: str
    index: int
    page_info: PageInfo | None = None
    embedding: list[float] | None = None
    entity_names: list[str] = field(default_factory=list)
    sentence_count: int = 0
    token_estimate: int = 0
    parent_chunk_id: str | None = None  # Set when using parent-child chunking
    is_parent: bool = False             # True for parent chunks in parent-child strategy
```

The `parent_chunk_id` and `is_parent` fields are only populated when the `parent_child` chunker is active.

---

### CommunityInfo

**Module:** `cognity-ai/models/knowledge.py`

```python
@dataclass
class CommunityInfo:
    community_id: str
    level: int
    entity_names: list[str] = field(default_factory=list)
    summary: str = ""
    title: str = ""
    parent_community: str | None = None
    rank: float = 0.0
    embedding: list[float] | None = None
```

---

### RetrievalResult

**Module:** `cognity-ai/models/retrieval.py`

```python
@dataclass
class RetrievalResult:
    content: str                       # Text content (triple, chunk text, or community summary)
    score: float                       # Retrieval score after RRF fusion and confidence boosts
    source: str                        # "graph" | "vector" | "community" | "vector_bridge" | "page"
    metadata: dict = field(default_factory=dict)
```

---

### SourceStatus

**Module:** `cognity-ai/models/knowledge.py`

```python
from enum import Enum

class SourceStatus(str, Enum):
    PENDING    = "pending"
    CONFIRMED  = "confirmed"
    DEPRECATED = "deprecated"
```

---

## 3. Configuration

**Module:** `cognity-ai/config/`

All configuration is expressed as dataclasses defined in `cognity-ai/config/base.py`. Provider-specific configs are in `cognity-ai/config/providers.py`. The top-level `LibraryConfig` aggregates all provider configs.

---

### LibraryConfig — Top-Level Fields

| Field | Type | Default | Valid Values |
|-------|------|---------|-------------|
| `rag_method` | `str` | `"hybrid_graph"` | `hybrid_graph`, `naive`, `vector_only`, `graph_only`, `parent_child`, `multi_query`, `microsoft_graphrag`, `adaptive` |
| `chunker` | `str` | `"sentence"` | `sentence`, `fixed`, `recursive`, `semantic`, `parent_child`, `hybrid` |
| `embedder` | `str` | `"gemini"` | `gemini`, `vertex_ai`, `openai`, `azure_openai`, `bedrock`, `cohere`, `sentence_transformers`, `ollama` |
| `vector_store` | `str` | `"chroma"` | `chroma`, `qdrant`, `pinecone`, `faiss`, `weaviate`, `milvus`, `pgvector`, `azure_search` |
| `graph_store` | `str` | `"neo4j"` | `neo4j`, `microsoft_graphrag`, `memgraph`, `arangodb`, `networkx`, `none` |
| `llm` | `str` | `"gemini"` | `gemini`, `vertex_ai`, `openai`, `azure_openai`, `anthropic`, `bedrock`, `cohere`, `ollama` |
| `extraction` | `str` | `"hybrid"` | `hybrid`, `nlp_only`, `llm_only` |
| `ocr` | `str` | `"gemini_vision"` | `gemini_vision`, `openai_vision`, `anthropic_vision`, `azure_vision`, `bedrock_vision`, `tesseract` |
| `page_index` | `str` | `"hybrid"` | `hybrid`, `regex`, `structural` |

Access provider sub-configs via attribute:

```python
from cognity_ai.config import LibraryConfig

cfg = LibraryConfig()
cfg.gemini.api_key = "AIza..."
cfg.neo4j.password = "secret"

lib = RAGLibrary(config=cfg)
```

---

### Provider Configs

#### GeminiConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` | `""` | Gemini API key |
| `model` | `str` | `"gemini-2.0-flash"` | Generation model ID |
| `embedding_model` | `str` | `"models/text-embedding-004"` | Embedding model ID |
| `temperature` | `float` | `0.1` | Generation temperature |
| `batch_embed_limit` | `int` | `100` | Max texts per embedding API call |
| `rpm_limit` | `int` | `15` | Rate limit (requests per minute) |

---

#### OpenAIConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` | `""` | OpenAI API key |
| `model` | `str` | `"gpt-4o"` | Generation model ID |
| `embedding_model` | `str` | `"text-embedding-3-small"` | Embedding model ID |
| `temperature` | `float` | `0.1` | Generation temperature |

---

#### AnthropicConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` | `""` | Anthropic API key |
| `model` | `str` | `"claude-3-5-sonnet-20241022"` | Model ID |
| `temperature` | `float` | `0.1` | Generation temperature |

---

#### AzureOpenAIConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint` | `str` | `""` | Azure OpenAI resource endpoint URL |
| `api_key` | `str` | `""` | Azure API key |
| `deployment_name` | `str` | `"gpt-4o"` | Chat completion deployment name |
| `api_version` | `str` | `"2024-02-01"` | API version string |
| `embedding_deployment` | `str` | `""` | Embedding deployment name (if separate) |

---

#### BedrockConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `region` | `str` | `"us-east-1"` | AWS region |
| `access_key_id` | `str` | `""` | AWS access key ID |
| `secret_access_key` | `str` | `""` | AWS secret access key |
| `model_id` | `str` | `"anthropic.claude-3-5-sonnet-20241022-v2:0"` | Bedrock model ID for generation |
| `embedding_model_id` | `str` | `"amazon.titan-embed-text-v2:0"` | Bedrock model ID for embeddings |

---

#### Neo4jConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uri` | `str` | `"bolt://localhost:7687"` | Neo4j connection URI |
| `user` | `str` | `"neo4j"` | Database user |
| `password` | `str` | `""` | Database password |
| `database` | `str` | `"neo4j"` | Target database name |

---

#### ChromaConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `persist_directory` | `str` | `"./chroma_store"` | On-disk persistence path |
| `collection_name` | `str` | — | Name for the chunk collection |
| `community_collection` | `str` | — | Name for the community summary collection |

---

#### NLPConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spacy_model` | `str` | `"en_core_web_trf"` | Primary spaCy model name |
| `fallback_model` | `str` | `"en_core_web_sm"` | Fallback if primary is unavailable |
| `semantic_chunk_sentences` | `int` | `5` | Sentences per semantic chunk |
| `semantic_chunk_overlap` | `int` | `1` | Overlapping sentences between adjacent chunks |

---

#### IngestionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hash_store_path` | `str` | `"./doc_hashes.json"` | Path for the SHA-256 hash store |
| `page_index_path` | `str` | `"./page_index.json"` | Path for the page/section index |
| `confidence_threshold` | `float` | `0.5` | Default pruning threshold |
| `confirmed_boost` | `float` | `1.5` | Score multiplier applied to confirmed-source results |
| `gemini_extraction_mode` | `str` | `"augment"` | `"augment"` (NLP-first) or `"full"` (LLM-only) |
| `max_gemini_chunks_per_doc` | `int` | `50` | Cap on LLM extraction calls per document |

---

## 4. Abstract Base Classes

These classes define the contracts that all built-in and custom plugin implementations must satisfy. Import them from their respective modules when writing plugins.

---

### BaseLoader

**Module:** `cognity-ai/loaders/base.py`

```python
from abc import ABC, abstractmethod
from cognity_ai.models import Document

class BaseLoader(ABC):

    @abstractmethod
    def load(self, path: str) -> list[Document]:
        """Load a file and return one or more Document objects."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return the list of file extensions this loader handles, e.g. ['.pdf']."""
        ...
```

Multiple `Document` objects may be returned when a single file contains logically separate units (e.g., sheets in a workbook, slides in a presentation).

---

### BaseOCR

**Module:** `cognity-ai/ocr/base.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path

class BaseOCR(ABC):

    @abstractmethod
    def ocr(self, image: str | bytes | Path) -> str:
        """Extract text from an image. Input may be a file path, raw bytes, or Path object."""
        ...

    @property
    def supports_multimodal(self) -> bool:
        """Return True if this OCR backend can handle multi-image inputs natively."""
        return False
```

---

### BaseChunker

**Module:** `cognity-ai/chunkers/base.py`

```python
from abc import ABC, abstractmethod
from cognity_ai.models import SemanticChunk, PageInfo

class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        pages: list[PageInfo] | None = None,
    ) -> list[SemanticChunk]:
        """Split text into chunks. Page boundaries are provided for page-aware chunkers."""
        ...
```

---

### BaseEmbedder

**Module:** `cognity-ai/embedders/base.py`

```python
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        task_type: str = "retrieval_document",
    ) -> list[list[float]]:
        """Embed a list of texts. Returns one embedding vector per input text."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string using the query task type."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors produced by this embedder."""
        ...
```

`task_type` values follow the Google embedding API convention (`"retrieval_document"`, `"retrieval_query"`) but implementations for other providers map these semantically.

---

### BaseVectorStore

**Module:** `cognity-ai/stores/vector/base.py`

```python
from abc import ABC, abstractmethod
from cognity_ai.models import SemanticChunk, CommunityInfo, RetrievalResult

class BaseVectorStore(ABC):

    @abstractmethod
    def upsert_chunks(self, chunks: list[SemanticChunk]): ...

    @abstractmethod
    def query_chunks(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict = None,
    ) -> list[RetrievalResult]: ...

    @abstractmethod
    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]: ...

    @abstractmethod
    def upsert_community(self, community: CommunityInfo): ...

    @abstractmethod
    def query_communities(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]: ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str): ...
```

`filters` in `query_chunks` is a store-specific dict for metadata filtering (e.g., `{"doc_id": "report"}` for Chroma, or a Qdrant filter object).

---

### BaseGraphStore

**Module:** `cognity-ai/stores/graph/base.py`

Defines 16 abstract methods covering the full graph lifecycle:

| Method | Description |
|--------|-------------|
| `upsert_entity(entity)` | Insert or update an entity node |
| `upsert_relation(relation)` | Insert or update a relation edge |
| `link_chunk_to_entities(chunk_id, doc_id, entity_names)` | Connect chunk references to entity nodes |
| `retrieve_subgraph(entity_names, hops, limit)` | BFS subgraph expansion from seed entities |
| `global_community_search(top_n)` | Return top community summaries by rank |
| `detect_communities()` | Run community detection; return raw cluster data |
| `get_chunks_for_entities(entity_names)` | Return chunk IDs linked to given entities (graph→vector bridge) |
| `upsert_doc_meta(doc_id, hash, source_name, status, stats)` | Store or update document metadata |
| `remove_doc_subgraph(doc_id)` | Delete all graph data for a document |
| `get_doc_status(doc_id)` | Return `"pending"` \| `"confirmed"` \| `"deprecated"` \| `None` |
| `confirm_source(doc_id)` | Boost triple confidences to 1.0 |
| `deprecate_source(doc_id)` | Halve triple confidences |
| `bulk_confirm(doc_ids)` | Batch confirm |
| `bulk_deprecate(doc_ids)` | Batch deprecate |
| `prune_low_confidence(threshold)` | Delete relations below threshold; return count |
| `health_report()` | Return knowledge base health dict |

---

### BaseGenerator

**Module:** `cognity-ai/generators/base.py`

```python
from abc import ABC, abstractmethod

class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Generate an answer to question given the retrieved context string."""
        ...
```

---

### BaseRetriever

**Module:** `cognity-ai/retrievers/base.py`

```python
from abc import ABC, abstractmethod
from cognity_ai.models import RetrievalResult

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Return ranked retrieval results for the query."""
        ...

    def query(self, question: str, top_k: int = 10) -> str:
        """Retrieve context and generate an answer. Default implementation calls self.retrieve()
        then passes context to the configured generator."""
        ...

    def query_with_sources(self, question: str, top_k: int = 10) -> dict:
        """Retrieve context, generate an answer, and return full source attribution."""
        ...
```

`query` and `query_with_sources` have default implementations in `BaseRetriever` that call `retrieve()` internally, so custom retrievers only need to implement `retrieve()`.

---

## 5. PluginRegistry

**Module:** `cognity-ai/registry.py`

`PluginRegistry` is a class with only class methods — no instantiation required. It is the shared registry that `RAGLibrary` consults during component construction. Plugins registered on `PluginRegistry` are immediately available to any `RAGLibrary` instance created afterward.

```python
from cognity_ai.registry import PluginRegistry

PluginRegistry.register_loader(".myext", MyLoader)
PluginRegistry.register_chunker("my_chunker", MyChunker)
PluginRegistry.register_embedder("my_embedder", MyEmbedder)
PluginRegistry.register_generator("my_llm", MyGenerator)
PluginRegistry.register_retriever("my_strategy", MyRetriever)
PluginRegistry.register_ocr("my_ocr", MyOCR)
```

| Class Method | Description |
|-------------|-------------|
| `register_loader(ext, cls)` | Register a loader for a file extension |
| `get_loader(ext) → type` | Retrieve the loader class for an extension |
| `register_chunker(name, cls)` | Register a chunker by name |
| `get_chunker(name) → type` | Retrieve a chunker class by name |
| `register_embedder(name, cls)` | Register an embedder by name |
| `get_embedder(name) → type` | Retrieve an embedder class by name |
| `register_generator(name, cls)` | Register a generator by name |
| `get_generator(name) → type` | Retrieve a generator class by name |
| `register_retriever(name, cls)` | Register a retriever by name |
| `get_retriever(name) → type` | Retrieve a retriever class by name |
| `register_ocr(name, cls)` | Register an OCR backend by name |
| `get_ocr(name) → type` | Retrieve an OCR backend class by name |
| `summary() → dict` | Return a dict listing all registered names per type |

`get_*` methods raise `KeyError` when the requested name is not registered.

---

## 6. Utility Modules

**Module:** `cognity-ai/utils/`

---

### HashStore

**Module:** `cognity-ai/utils/hash.py`

#### `content_hash(text) → str`

Compute the SHA-256 hex digest of a UTF-8 string. Used internally to detect unchanged documents and skip re-ingestion.

```python
from cognity_ai.utils.hash import content_hash
digest = content_hash("hello world")
```

---

#### `HashStore(path)`

JSON-backed persistent store mapping `doc_id → SHA-256 hash`. Used by the ingestion pipeline to implement change detection.

| Method | Description |
|--------|-------------|
| `get(doc_id) → str \| None` | Return the stored hash for a document, or `None` if absent |
| `set(doc_id, hash)` | Store or update the hash for a document |
| `remove(doc_id)` | Delete the hash entry for a document |
| `is_unchanged(doc_id, text) → bool` | Return `True` if `content_hash(text)` matches the stored hash |
| `all_doc_ids() → set[str]` | Return all doc IDs currently in the store |

```python
from cognity_ai.utils.hash import HashStore

store = HashStore("./doc_hashes.json")
store.set("doc1", content_hash("some text"))
store.is_unchanged("doc1", "some text")   # True
store.is_unchanged("doc1", "other text")  # False
```

---

### reciprocal_rank_fusion

**Module:** `cognity-ai/utils/rrf.py`

```python
def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievalResult],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[RetrievalResult]:
```

Merge N ranked result lists using the Reciprocal Rank Fusion formula. Results are deduplicated by the first 120 characters of their `content` field before scoring.

The RRF score for a result appearing at rank `r` in list `i` is:

```
score_i(r) = weight_i / (k + r)
```

Final scores are the sum across all lists. Results are returned sorted descending by final score.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `*ranked_lists` | `list[RetrievalResult]` | — | Any number of ranked result lists |
| `weights` | `list[float] \| None` | `None` | Per-list weights; uniform if `None` |
| `k` | `int` | `60` | RRF damping constant |

```python
from cognity_ai.utils.rrf import reciprocal_rank_fusion

fused = reciprocal_rank_fusion(graph_results, vector_results, weights=[0.6, 0.4])
```

---

### Token Counter

**Module:** `cognity-ai/utils/token_counter.py`

#### `estimate_tokens(text, method="word") → int`

Estimate the token count of a string. `method="word"` uses a fast word-split heuristic (suitable for cost estimation). Other methods may be added by future embedder integrations.

---

#### `split_to_token_limit(text, max_tokens, method="word") → list[str]`

Split a long string into segments each containing at most `max_tokens` estimated tokens. Useful for chunking large documents before LLM calls that have context-window limits.

---

## 7. PDF Utilities

**Module:** `cognity-ai/loaders/pdf_utils.py`

Standalone helper functions for working with PDF files. These are used internally by the PDF loader but are also importable directly.

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_tables` | `(path) → list[DataFrame]` | Extract tables from each page as a list of pandas DataFrames via pdfplumber |
| `extract_images` | `(path) → list[bytes]` | Extract raw image bytes from all pages |
| `extract_metadata` | `(path) → dict` | Return document metadata: `author`, `title`, `creation_date`, `page_count` |
| `slice_pages` | `(path, start, end) → bytes` | Return a subset of pages as a new PDF byte string |
| `merge_pdfs` | `(paths) → bytes` | Merge multiple PDF files into a single PDF byte string |
| `pdf_to_images` | `(path, dpi=150) → list[bytes]` | Render each page as a JPEG image (for full-page LLM OCR workflows) |

```python
from cognity_ai.loaders.pdf_utils import extract_tables, pdf_to_images

tables = extract_tables("report.pdf")          # list of DataFrames, one per table found
images = pdf_to_images("report.pdf", dpi=200)  # list of JPEG bytes, one per page
```

---

## 8. Backward Compatibility

**Module:** `hybrid_rag.main`

The legacy `build_pipeline()` function continues to work but is deprecated. It emits a `DeprecationWarning` on every call.

```python
def build_pipeline(config=None) -> dict:
    ...
```

**Returns** the legacy component dict:

```python
{
    "config": Config,
    "nlp": NLPProcessor,
    "gemini": GeminiExtractor,
    "graph": GraphManager,
    "vector": VectorManager,
    "page_idx": PageIndex,
    "pipeline": IngestionPipeline,
    "retriever": HybridRetriever,
    "updater": KnowledgeUpdater,
}
```

**Migrate to `RAGLibrary`:**

```python
# Before (deprecated)
from hybrid_rag.main import build_pipeline
c = build_pipeline()
c["pipeline"].ingest(doc_id="d1", text="...")
answer = c["retriever"].query("What is X?")

# After
from cognity_ai import RAGLibrary
lib = RAGLibrary(gemini_api_key="...", neo4j_password="...")
lib.ingest_text("...", doc_id="d1")
answer = lib.query("What is X?")
```

The internal classes (`NLPProcessor`, `GeminiExtractor`, `GraphManager`, `VectorManager`, `HybridRetriever`, `KnowledgeUpdater`) are no longer part of the public API. Access equivalent functionality through `RAGLibrary` or the abstract base classes for plugin development.
