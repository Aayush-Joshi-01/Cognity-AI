# API Reference

Complete code documentation for every module, class, and public method.

---

## Table of Contents

- [main.py — Entry Point](#mainpy)
- [config.py — Configuration](#configpy)
- [models.py — Data Models](#modelspy)
- [nlp_processor.py — Local NLP Pipeline](#nlp_processorpy)
- [gemini_extractor.py — LLM Layer](#gemini_extractorpy)
- [graph_manager.py — Neo4j Operations](#graph_managerpy)
- [vector_manager.py — ChromaDB Operations](#vector_managerpy)
- [page_index.py — Page/Section Index](#page_indexpy)
- [ingestion.py — Ingestion Pipeline](#ingestionpy)
- [hybrid_retriever.py — 4-Channel Retriever](#hybrid_retrieverpy)
- [knowledge_updater.py — Lifecycle Management](#knowledge_updaterpy)

---

## `main.py`

### `build_pipeline(config=None) → dict`

Factory function that wires all components together.

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `config` | `Config \| None` | `None` | Configuration object. Uses defaults if `None`. |

**Returns:** Dictionary of named components:
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

**Usage:**
```python
c = build_pipeline()
c["pipeline"].ingest(doc_id="d1", text="...")
answer = c["retriever"].query("What is X?")
```

---

## `config.py`

All configuration is expressed as dataclasses. Every field has a sensible default and can be overridden via constructor or environment variable (where noted).

### `Neo4jConfig`

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `uri` | `str` | `"neo4j+s://xxxxx..."` | `NEO4J_URI` |
| `user` | `str` | `"neo4j"` | `NEO4J_USER` |
| `password` | `str` | `""` | `NEO4J_PASSWORD` |
| `database` | `str` | `"neo4j"` | `NEO4J_DATABASE` |

### `GeminiConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | `str` | `""` (env) | `GEMINI_API_KEY` |
| `model` | `str` | `"gemini-2.0-flash"` | Generation model |
| `embedding_model` | `str` | `"models/text-embedding-004"` | Embedding model |
| `temperature` | `float` | `0.1` | Generation temperature |
| `extraction_temperature` | `float` | `0.0` | Extraction temperature (deterministic) |
| `batch_embed_limit` | `int` | `100` | Max texts per embed API call |
| `rpm_limit` | `int` | `15` | Rate limit (requests/minute) |

### `ChromaConfig`

| Field | Type | Default |
|-------|------|---------|
| `persist_directory` | `str` | `"./chroma_store"` |
| `collection_name` | `str` | `"hybrid_rag_chunks"` |
| `community_collection` | `str` | `"community_summaries"` |

### `NLPConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spacy_model` | `str` | `"en_core_web_trf"` | Primary spaCy model |
| `fallback_model` | `str` | `"en_core_web_sm"` | Fallback if primary unavailable |
| `min_entity_freq` | `int` | `1` | Minimum entity occurrences to keep |
| `dependency_relations` | `list[str]` | `["nsubj", "dobj", ...]` | Dependency arcs to extract |
| `semantic_chunk_sentences` | `int` | `5` | Sentences per semantic chunk |
| `semantic_chunk_overlap` | `int` | `1` | Sentence overlap between chunks |

### `GraphRAGConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `leiden_resolution` | `float` | `1.0` | Higher = more granular communities |
| `max_community_levels` | `int` | `3` | Hierarchical depth for Leiden |
| `community_summary_max_tokens` | `int` | `300` | Max tokens in community summary |
| `local_search_top_k` | `int` | `10` | Default graph local search limit |
| `global_search_top_communities` | `int` | `5` | Communities in global search |

### `IngestionConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hash_store_path` | `str` | `"./doc_hashes.json"` | Persistent hash store |
| `page_index_path` | `str` | `"./page_index.json"` | Persistent page index |
| `confidence_threshold` | `float` | `0.5` | Pruning cutoff |
| `confirmed_boost` | `float` | `1.5` | Score multiplier for confirmed sources |
| `use_local_nlp_first` | `bool` | `True` | Enable NLP-first extraction |
| `gemini_extraction_mode` | `str` | `"augment"` | `"augment"` or `"full"` |
| `max_gemini_chunks_per_doc` | `int` | `50` | Cap API calls per document |
| `cache_embeddings` | `bool` | `True` | Enable embedding cache |

---

## `models.py`

Pydantic v2 data models used across the system.

### `Entity`

```python
class Entity(BaseModel):
    name: str                          # Canonical name (title case)
    entity_type: str                   # "Person", "Organization", "Concept", etc.
    description: str = ""              # One-line description
    properties: dict = {}              # Arbitrary key-value metadata
    source_id: str = ""                # Origin document ID
    confidence: float = 1.0            # 0.0–1.0
    extraction_method: str = "nlp"     # "nlp" | "llm" | "merged" | "nlp_np" | "nlp_svo"
    mentions: int = 1                  # Cross-chunk frequency count
```

### `Relation`

```python
class Relation(BaseModel):
    source_entity: str                 # Source entity name
    relation_type: str                 # UPPER_SNAKE_CASE relation label
    target_entity: str                 # Target entity name
    description: str = ""              # Natural language description
    properties: dict = {}
    source_id: str = ""
    confidence: float = 1.0
    extraction_method: str = "nlp"
    weight: float = 1.0                # Frequency weight (accumulates on dedup)
```

### `ExtractionResult`

```python
class ExtractionResult(BaseModel):
    entities: list[Entity] = []
    relations: list[Relation] = []
```

### `PageInfo`

```python
class PageInfo(BaseModel):
    page_num: int
    section: str = ""
    start_char: int = 0
    end_char: int = 0
    heading: str = ""
```

### `SemanticChunk`

```python
class SemanticChunk(BaseModel):
    chunk_id: str                      # "{doc_id}__chunk_{index}"
    doc_id: str
    text: str
    index: int
    page_info: Optional[PageInfo] = None
    embedding: Optional[list[float]] = None
    entity_names: list[str] = []       # Entities mentioned in this chunk
    sentence_count: int = 0
    token_estimate: int = 0
```

### `CommunityInfo`

```python
class CommunityInfo(BaseModel):
    community_id: str
    level: int
    entity_names: list[str] = []
    summary: str = ""
    title: str = ""
    parent_community: Optional[str] = None
    rank: float = 0.0
    embedding: Optional[list[float]] = None
```

### `RetrievalResult`

```python
class RetrievalResult(BaseModel):
    content: str                       # Text content (triple, chunk, or summary)
    score: float                       # Retrieval score (post-RRF, post-boost)
    source: str                        # "graph" | "vector" | "community" | "vector_bridge" | "page"
    metadata: dict = {}                # Source-specific metadata
```

### `SourceStatus` (Enum)

```python
class SourceStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"
```

---

## `nlp_processor.py`

### `class NLPProcessor`

Local NLP pipeline using spaCy. Zero API cost.

#### `__init__(config: NLPConfig)`

Loads spaCy model (primary, then fallback). Auto-downloads fallback if missing.

#### `process(text: str, source_id: str = "") → ExtractionResult`

Full NLP extraction pipeline. Runs NER → dependency relations → SVO triples → coreference resolution → noun phrase entities → deduplication.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `text` | `str` | Input text to extract from |
| `source_id` | `str` | Document ID for provenance tracking |

**Returns:** `ExtractionResult` with deduplicated entities and relations.

#### `semantic_chunk(text: str, doc_id: str, pages: list[PageInfo] | None = None) → list[SemanticChunk]`

Sentence-boundary-aware chunking with entity overlap tracking.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `text` | `str` | Full document text |
| `doc_id` | `str` | Document identifier |
| `pages` | `list[PageInfo] \| None` | Page boundaries for page-aware chunking |

**Returns:** List of `SemanticChunk` objects with `entity_names` populated.

#### `detect_pages(text: str) → list[PageInfo]` (static)

Detects page/section boundaries via regex patterns (form feeds, separators, headings).

**Returns:** List of `PageInfo` with character offset ranges.

---

## `gemini_extractor.py`

### `class GeminiExtractor`

Gemini API layer. Handles augmentation, descriptions, summarization, and embeddings. All methods include rate limiting and retry logic.

#### `__init__(config: GeminiConfig)`

Configures Gemini client. Creates two model instances: one with JSON output (for extraction) and one with text output (for descriptions).

#### `augment_extraction(text: str, existing: ExtractionResult, source_id: str = "") → ExtractionResult`

Finds entities/relations that NLP missed. Input prompt includes already-extracted items to avoid duplication.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `text` | `str` | Chunk text (capped at 3000 chars for cost) |
| `existing` | `ExtractionResult` | NLP-extracted entities/relations |
| `source_id` | `str` | Document ID |

**Returns:** `ExtractionResult` with only NEW findings.

**Retries:** 3 attempts with exponential backoff (1s–10s).

#### `generate_entity_description(entity_name: str, entity_type: str, context_snippets: list[str]) → str`

Generates a one-line description for an entity using surrounding context.

#### `summarize_community(entity_names: list[str], relation_descriptions: list[str]) → dict`

Generates a community summary for GraphRAG.

**Returns:** `{"title": "...", "summary": "..."}`

#### `embed_batch(texts: list[str], task_type: str = "retrieval_document") → list[list[float]]`

Batch embedding with automatic chunking to respect API limits.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `texts` | `list[str]` | Texts to embed |
| `task_type` | `str` | `"retrieval_document"` or `"retrieval_query"` |

**Returns:** List of embedding vectors.

#### `embed_query(query: str) → list[float]`

Single query embedding with `retrieval_query` task type.

---

## `graph_manager.py`

### `class GraphManager`

Neo4j CRUD, community detection, and retrieval operations.

#### `__init__(neo4j_config: Neo4jConfig, graphrag_config: GraphRAGConfig)`

Opens driver connection, creates indexes on `Entity.name`, `Entity.source_id`, `DocumentMeta.doc_id`, `Community.community_id`, `ChunkRef.chunk_id`.

#### `close()`

Closes the Neo4j driver connection.

#### `upsert_entity(entity: Entity)`

MERGE on entity name. Higher confidence wins for type/confidence fields. Descriptions keep the longer version. Mentions accumulate. Dynamic secondary label added from `entity_type`.

#### `upsert_relation(relation: Relation)`

MERGE on `(source_entity)-[TYPE]->(target_entity)`. Higher confidence wins. Weight accumulates. Longer description wins.

#### `link_chunk_to_entities(chunk_id: str, doc_id: str, entity_names: list[str])`

Creates `ChunkRef` node and `MENTIONED_IN` edges from entities to the chunk. This is the graph↔vector bridge.

#### `upsert_doc_meta(doc_id, content_hash, source_name, status, stats=None)`

Stores/updates document metadata node.

#### `remove_doc_subgraph(doc_id: str)`

Deletes all chunk refs, relations, and orphan entities associated with a document.

#### `detect_communities() → list[dict]`

Runs Leiden (or Louvain fallback) community detection via Neo4j GDS. Returns list of `{"community_id": int, "members": list[str], "size": int}`.

**Requires:** Neo4j GDS plugin.

#### `get_community_entities(community_id: int) → list[dict]`

Returns all entities and intra-community relations for a given community.

#### `store_community_summary(community: CommunityInfo)`

Persists community summary as a `Community` node and links member entities via `BELONGS_TO_COMMUNITY`.

#### `retrieve_subgraph(entity_names: list[str], hops: int = 2, limit: int = 20) → list[RetrievalResult]`

BFS subgraph expansion from seed entities using APOC's `subgraphAll`. Excludes structural edges (`MENTIONED_IN`, `BELONGS_TO_COMMUNITY`). Returns triples as `RetrievalResult` with source="graph".

#### `retrieve_entity_context(entity_name: str) → list[RetrievalResult]`

Complete local neighborhood for a single entity: description + all incoming/outgoing relations.

#### `global_community_search(top_n: int = None) → list[RetrievalResult]`

Returns community summaries ranked by `rank` (entity count proportion). source="community".

#### `get_chunks_for_entities(entity_names: list[str]) → list[str]`

Returns chunk IDs linked to given entities via `MENTIONED_IN` edges. Powers the graph→vector bridge.

#### `confirm_source(doc_id: str)`

Sets status to "confirmed", boosts all triple confidences to 1.0.

#### `deprecate_source(doc_id: str)`

Sets status to "deprecated", halves all triple confidences.

#### `get_doc_status(doc_id: str) → str | None`

Returns "pending", "confirmed", "deprecated", or None.

#### `prune_low_confidence(threshold: float = 0.5) → int`

Deletes all relations with `confidence < threshold`. Returns count of pruned relations.

---

## `vector_manager.py`

### `class VectorManager`

ChromaDB operations with dual collections (chunks + communities).

#### `__init__(config: ChromaConfig)`

Creates persistent ChromaDB client. Initializes two collections with cosine distance.

#### `upsert_chunks(chunks: list[SemanticChunk])`

Batch upserts chunks with embeddings and metadata (doc_id, page_num, section, heading, entity_names as `|`-delimited string, sentence_count, token_estimate).

#### `delete_by_doc(doc_id: str)`

Removes all chunks belonging to a document. Handles ChromaDB's `where`-based delete with fallback to ID-based delete.

#### `query_chunks(embedding, top_k=10, filter_doc_ids=None, filter_page=None) → list[RetrievalResult]`

Cosine similarity search on chunk collection. Supports filtering by document IDs and/or page number.

**Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `embedding` | `list[float]` | Query embedding vector |
| `top_k` | `int` | Max results |
| `filter_doc_ids` | `list[str] \| None` | Restrict to specific documents |
| `filter_page` | `int \| None` | Restrict to specific page number |

#### `query_by_chunk_ids(chunk_ids: list[str]) → list[RetrievalResult]`

Direct retrieval by chunk IDs. Used by the graph→vector bridge. Returns results with base score of 0.8, source="vector_bridge".

#### `upsert_community(community: CommunityInfo)`

Upserts a community summary into the community collection with embedding and metadata.

#### `query_communities(embedding, top_k=5) → list[RetrievalResult]`

Cosine similarity search on community summaries. source="community".

#### `count() → dict`

Returns `{"chunks": int, "communities": int}`.

---

## `page_index.py`

### `class PageIndex`

Persistent JSON-backed page/section index.

#### `__init__(path: str = "./page_index.json")`

#### `store(doc_id: str, pages: list[PageInfo])`

Saves page boundaries for a document.

#### `get(doc_id: str) → list[PageInfo]`

Retrieves page boundaries.

#### `remove(doc_id: str)`

Removes page data for a document.

#### `get_page_text(doc_id: str, page_num: int, full_text: str) → str | None`

Extracts text for a specific page using stored character offsets.

#### `get_section(doc_id: str, heading: str) → PageInfo | None`

Finds a section by heading substring match.

#### `summary(doc_id: str) → dict`

Returns `{"doc_id": str, "page_count": int, "sections": [{"page": int, "heading": str}]}`.

---

## `ingestion.py`

### `class IngestionPipeline`

Orchestrates the full ingestion flow: hash check → page detection → chunking → NLP extraction → Gemini augmentation → deduplication → embedding → graph upsert → vector upsert.

#### `__init__(nlp, gemini, graph, vector, page_idx, config)`

Takes all component instances.

#### `ingest(doc_id: str, text: str, source_name: str = "", status: str = "pending") → dict`

Ingest a single document. Returns stats dict:

```python
# If unchanged:
{"doc_id": "d1", "status": "skipped", "reason": "unchanged"}

# If processed:
{"doc_id": "d1", "status": "ingested", "pages": 3, "chunks": 12, "entities": 28, "relations": 15}
```

**Full flow:**
1. SHA-256 hash check → skip if unchanged
2. Clear stale data if doc existed before
3. Detect pages/sections
4. Semantic chunking (sentence-boundary)
5. NLP extraction per chunk (NER + SVO + deps + coref + noun phrases)
6. Gemini augmentation per chunk (if `mode="augment"`)
7. Cross-chunk entity deduplication (higher confidence wins, mentions accumulate)
8. Cross-chunk relation deduplication (higher confidence wins, weight accumulates)
9. Batch embed all chunks
10. Upsert entities + relations + chunk-entity links to Neo4j
11. Upsert doc metadata to Neo4j
12. Upsert chunks to ChromaDB
13. Update hash store

#### `ingest_batch(documents: list[dict]) → list[dict]`

Batch ingest. Each dict must have `doc_id` and `text`. Optional: `source_name`, `status`.

#### `build_communities() → list[CommunityInfo]`

Post-ingestion step. Runs Leiden community detection, generates summaries via Gemini, embeds summaries, stores in both Neo4j and ChromaDB.

#### `remove_document(doc_id: str)`

Full removal from graph, vector store, page index, and hash store.

#### `sync(current_doc_ids: set[str]) → list[str]`

Garbage collection. Removes docs in hash store but not in `current_doc_ids`. Returns list of removed doc IDs.

---

## `hybrid_retriever.py`

### `class HybridRetriever`

4-channel hybrid retrieval with RRF fusion and Gemini generation.

#### `__init__(nlp, gemini, graph, vector, config)`

#### `retrieve(query: str, top_k=10, graph_hops=2, use_communities=True, use_bridge=True) → list[RetrievalResult]`

Pure retrieval (no generation). Returns fused and ranked results from all active channels.

**Flow:**
1. NLP extracts seed entities from query (free). Falls back to noun phrases if no named entities found.
2. Channel 1: Graph subgraph traversal from seed entities
3. Channel 2: Vector cosine similarity on query embedding
4. Channel 3: Community summaries (graph-ranked + vector-similarity)
5. Channel 4: Graph→Vector bridge (entity-linked chunks)
6. RRF fusion with channel weights
7. Confirmed-source boost / deprecated-source penalty

**Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | — | Natural language question |
| `top_k` | `int` | `10` | Max results after fusion |
| `graph_hops` | `int` | `2` | BFS depth for graph traversal |
| `use_communities` | `bool` | `True` | Enable community channel |
| `use_bridge` | `bool` | `True` | Enable graph→vector bridge |

#### `query(question: str, top_k=10, graph_hops=2) → str`

Full RAG: retrieve + generate. Returns Gemini's answer as a string.

#### `query_with_sources(question: str, top_k=10) → dict`

Full RAG with source attribution:

```python
{
    "answer": str,                    # Gemini's generated answer
    "sources": {
        "graph": [dict, ...],         # Graph result metadata
        "vector": [dict, ...],        # Vector result metadata
        "community": [dict, ...],     # Community result metadata
    },
    "retrieval_scores": [             # Top 8 results with scores
        {"content": str, "score": float, "channel": str},
    ],
    "seed_entities": [str, ...],      # Entities extracted from query
}
```

---

## `knowledge_updater.py`

### `class KnowledgeUpdater`

Source lifecycle, conflict detection, pruning, and health reporting.

#### `__init__(graph: GraphManager, config: Config)`

#### `confirm_source(doc_id: str)`

Marks source as confirmed. All triples get confidence 1.0.

#### `deprecate_source(doc_id: str)`

Marks source as deprecated. All triple confidences halved.

#### `bulk_confirm(doc_ids: list[str])`

Batch confirm.

#### `bulk_deprecate(doc_ids: list[str])`

Batch deprecate.

#### `detect_conflicts(entity_name: str) → list[dict]`

Finds conflicting relations for an entity across sources.

**Returns:**
```python
[{
    "entity": "Anthropic",
    "relation": "RAISED",
    "versions": [
        {"src": "Anthropic", "rel": "RAISED", "tgt": "$7 Billion", "source_id": "doc_old", "confidence": 0.5},
        {"src": "Anthropic", "rel": "RAISED", "tgt": "$10 Billion", "source_id": "doc_new", "confidence": 1.0},
    ]
}]
```

#### `prune_low_confidence(threshold: float = None) → int`

Removes relations below threshold. Returns count pruned.

#### `get_source_stats() → list[dict]`

Returns per-document stats: doc_id, source name, status, triple count, chunk/entity/relation counts.

#### `health_report() → dict`

Overall knowledge base health:
```python
{
    "entities": int,
    "relations": int,
    "documents": int,
    "confirmed_sources": int,
    "communities": int,
    "avg_confidence": float,
}
```
