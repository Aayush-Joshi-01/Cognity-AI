# Changelog

All notable changes to cognity-ai will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.1.0] - 2026-04-05

### Added — Trie Data Structure (`cognity_ai/utils/trie.py`)

- **`TrieNode`** — internal trie node with `children`, `is_end`, and `original` slots
- **`Trie`** — full prefix tree with:
  - `insert`, `search`, `starts_with`, `delete`
  - `words_with_prefix(prefix, max_results)` — DFS autocomplete
  - `autocomplete(prefix)` — alias for `words_with_prefix`
  - `all_words()` — BFS traversal returning all stored words
  - `count_words()`, `__len__`, `__contains__`, `__iter__`
  - `longest_prefix_match(query)` — longest stored prefix of *query*
  - `bfs_nodes()` / `dfs_nodes()` — structural traversal for inspection/debugging
- **`EntityTrie(Trie)`** — case-insensitive entity lookup, preserves original casing:
  - `insert_entity(name)` — lowercased key, original-case stored in terminal node
  - `search_entities(prefix, max_results)` — returns original-case matches
  - `delete_entity(name)`
- **`NetworkXStore`** now maintains an `EntityTrie` alongside the NetworkX graph:
  - `upsert_entity()` auto-inserts into the trie
  - `retrieve_subgraph()` and `retrieve_entity_context()` replaced O(n) linear scans with O(k) trie lookups
  - New `suggest_entities(prefix, max_results=10)` public method
- **`BaseGraphStore`** — added default `suggest_entities()` method (no-op stub for backends without a trie)
- **`BasePageIndex`** — trie-accelerated `get_section()`:
  - `_build_heading_index(doc_id, pages)` builds a per-document `Trie` of lowercased headings
  - `get_section()` uses `words_with_prefix` for O(k) lookup, falls back to linear scan if trie not yet built
  - Hooked into `store()` / `remove()` of `RegexPageIndex`, `StructuralPageIndex`, `HybridPageIndex`
- **`RAGLibrary.suggest_entities(prefix, max_results=10)`** — delegates to graph store

### Added — AI Observability & Token Tracking (`cognity_ai/observability/`)

- **`models.py`** — Pydantic event models:
  - `TokenUsage` — prompt/completion/total tokens with `source` field (`"native"` | `"tiktoken"` | `"estimate"`) and `__add__` for aggregation
  - `GenerationEvent`, `RetrievalEvent`, `EmbedEvent` — timestamped event records with latency
- **`token_tracker.py`** — pluggable token counting chain:
  - `BaseTokenCounter` ABC — extend to add custom counters (OpenTelemetry, Langfuse, etc.)
  - `NativeTokenCounter` — extracts usage from raw API response objects; per-provider extractors for Gemini (`usage_metadata`), OpenAI (`usage`), Anthropic (`message.usage`), Cohere (`resp.meta.tokens`), Ollama (`eval_count` from JSON), Bedrock (decoded body `inputTokenCount`)
  - `TiktokenCounter` — uses `tiktoken` when installed and model is known to it; returns -1 (falls through) for unknown models
  - `EstimateCounter` — word-count heuristic; always available
  - `TokenTracker` — priority chain: native → tiktoken → estimate; accepts `extra_counters` for injection
- **`base_observer.py`** — `BaseObserver` ABC with `on_generation()`, `on_retrieval()`, `on_embed()` hooks; all no-ops by default so subclasses only override what they need. Extension point for OpenTelemetry, Prometheus, Langfuse, Datadog, etc.
- **`noop_observer.py`** — `NoopObserver` — zero-overhead default
- **`logging_observer.py`** — `LoggingObserver` — emits each event as JSON to `cognity_ai.observability` logger
- **`collector.py`** — `ObservabilityCollector`:
  - Fan-out to all registered observers; observer errors are silently caught (never crash the pipeline)
  - In-memory ring buffer (`deque(maxlen=max_event_buffer)`) for recent event inspection
  - Aggregate stats: `get_summary()` → total calls, total tokens, buffered event count
  - `add_observer()`, `remove_observer()`, `reset()`, `recent_events(n)`
  - `enabled=False` makes all emits no-ops
- **`ObservabilityConfig`** dataclass in `cognity_ai/config/providers.py`:
  - `enabled: bool = True`, `observer: str = "noop"`, `log_level: str = "INFO"`, `max_event_buffer: int = 1000`
- **`LibraryConfig`** and **`MinimalLibraryConfig`** now include `observability: ObservabilityConfig` field
- **All generators instrumented** — `BaseGenerator` gains `_collector` slot and `set_collector()` / `_emit_generation()`:
  - `GeminiGenerator.generate()` and `generate_rag()` — extracts `resp.usage_metadata`
  - `OpenAIGenerator.generate()` — extracts `resp.usage`
  - `AnthropicGenerator.generate()` — extracts `message.usage`
  - `CohereGenerator.generate()` — extracts `resp.meta.tokens`
  - `OllamaGenerator.generate()` — extracts `prompt_eval_count`/`eval_count` from JSON
  - `BedrockGenerator.generate()` — extracts `inputTokenCount`/`outputTokenCount` from decoded body
- **`RAGLibrary`** wires observability on construction:
  - Accepts `observer=` (any `BaseObserver` instance) and `observability_config=`
  - Attaches collector to generator via `set_collector()`
  - `rag.observability` property — returns the `ObservabilityCollector`
  - `rag.token_summary()` — shortcut for `collector.get_summary()`

### Added — Test Suite

- **`pytest.ini`** — `testpaths = tests`, `pythonpath = .`, verbose output
- **`tests/conftest.py`** — credential flags, availability flags, shared fixtures (`tmp_dir`, `make_embedding`, etc.)
- **`MinimalLibraryConfig`** — bare-minimum preset (FAISS + NetworkX + fixed chunker + llm_only extraction); safe to instantiate without any external services
- New test files:
  - `tests/test_trie.py` — 41 tests covering all Trie and EntityTrie operations
  - `tests/test_observability.py` — 39 tests for token tracking, observers, collector, generator integration
  - `tests/test_config.py`, `tests/test_models.py`, `tests/test_utils.py` — data model and config tests
  - `tests/test_loaders.py`, `tests/test_chunkers.py` — content processing tests
  - `tests/test_vector_stores.py`, `tests/test_graph_stores.py` — store backend tests
  - `tests/test_embedders.py`, `tests/test_generators.py` — provider tests (skip without credentials)
  - `tests/test_factory.py` — structural wiring tests using `unittest.mock.patch`
- **Full suite result: 316 passed, 49 skipped** (skips are intentional — cloud providers without credentials)

### Changed

- All generators now time API calls (wall-clock ms) and emit `GenerationEvent` to the attached collector
- `NetworkXStore` entity lookups are now O(k) via `EntityTrie` instead of O(n) linear scan
- `BasePageIndex.get_section()` uses per-document heading trie when available
- License changed from **MIT** to **Apache 2.0** — see [LICENSE](LICENSE)
- `pyproject.toml` version bumped from `2.0.1` → `2.1.0`
- New exports in `cognity_ai/__init__.py`: `ObservabilityCollector`, `BaseObserver`, `LoggingObserver`, `TokenUsage`, `GenerationEvent`, `RetrievalEvent`, `EmbedEvent`, `ObservabilityConfig`, `Trie`, `EntityTrie`

---

## [2.0.0] - 2026-03-19

### Added — Core Library
- **`cognity-ai` package** — complete modular rewrite of `hybrid_rag` into a provider-agnostic library
- **`RAGLibrary` facade** — single unified entry point for all RAG operations: ingest, query, lifecycle
- **Plugin registry** — `PluginRegistry` for registering custom loaders, embedders, generators, retrievers
- **`ComponentFactory`** — auto-wires all components from string keys with smart fallback logic
- **`LibraryConfig`** — unified configuration dataclass replacing per-component configs

### Added — File Loaders
- **PDF loader** — pdfplumber primary, pypdf + pdfminer fallbacks; page-aware; embedded image extraction
- **PDF utilities** — `extract_tables()`, `extract_images()`, `extract_metadata()`, `slice_pages()`, `merge_pdfs()`, `pdf_to_images()`
- **DOCX loader** — python-docx; headings, tables, embedded images
- **Excel loader** — openpyxl + pandas; per-sheet text; formula values
- **PowerPoint loader** — python-pptx; slide text, speaker notes, embedded images
- **CSV/TSV loader** — pandas; auto-detect delimiter
- **HTML loader** — beautifulsoup4; tag stripping, heading extraction
- **JSON/YAML loader** — recursive key-value text representation
- **Image loader** — routes to OCR subsystem; supports JPEG, PNG, BMP, TIFF, WebP, GIF
- **Directory ingestion** — `ingest_dir()` for recursive multi-format folder ingestion

### Added — OCR Subsystem
- **Gemini Vision OCR** (default) — Gemini 2.0 Flash multimodal; complex layouts, tables, handwriting
- **OpenAI Vision OCR** — GPT-4o vision API
- **Anthropic Vision OCR** — Claude 3.5 Sonnet vision API
- **Azure Vision OCR** — Azure OpenAI GPT-4o vision
- **Bedrock Vision OCR** — AWS Bedrock Claude/Titan multimodal
- **Tesseract OCR** — local pytesseract (offline fallback)
- **OCR fallback chain** — auto-downgrades through providers if one fails
- **Embedded image handling** — images inside DOCX/PPTX/PDF extracted and OCR'd inline

### Added — LLM Providers
- **Vertex AI** — Google Vertex AI Gemini (generator + embedder)
- **Azure OpenAI** — Azure-deployed GPT-4o (generator + embedder)
- **Anthropic** — Claude 3.5/3.7 Sonnet generator (embedder auto-switches to sentence_transformers)
- **AWS Bedrock** — Claude, Titan, Llama, Mistral via boto3 (generator + Titan V2 embedder)
- **Cohere** — Command R+ generator + embed-english-v3.0 embedder
- **Ollama** — local Ollama REST API (generator + embedder)
- **OpenAI** — GPT-4o generator + text-embedding-3-small/large embedder

### Added — Vector Stores
- **Qdrant** — local or cloud; HNSW, named vectors
- **Pinecone** — pinecone-client v3 serverless + pod
- **FAISS** — faiss-cpu/gpu; flat or HNSW index; fastest local search
- **Weaviate** — weaviate-client v4; built-in BM25 + vector hybrid
- **Milvus** — pymilvus; HNSW COSINE; Zilliz cloud compatible
- **pgvector** — psycopg2 + PostgreSQL pgvector extension; IVFFlat index
- **Azure AI Search** — azure-search-documents; HNSW vector search

### Added — Graph Stores
- **Microsoft GraphRAG** — wraps official `microsoft/graphrag` library; local/global search
- **Memgraph** — open-source Neo4j-compatible Cypher; MAGE community detection
- **ArangoDB** — python-arango; AQL graph traversal; multi-model
- **NetworkX** — in-memory Python graph; greedy modularity communities; zero deps

### Added — Chunking Strategies
- **Fixed chunker** — configurable word-count chunks with overlap
- **Recursive chunker** — LangChain-style recursive character splitting
- **Semantic chunker** — groups sentences by embedding cosine similarity
- **Parent-child chunker** — small child chunks (retrieval) + large parent chunks (context)
- **Hybrid chunker** — sentence primary + optional semantic regrouping

### Added — Page Index Subsystem
- **Structural page index** — uses loader-provided PDF page numbers, slide numbers, sheet names
- **Hybrid page index** (default) — structural primary + regex fallback

### Added — Retrievers
- **Naive retriever** — pure vector cosine similarity
- **Vector-only retriever** — vector + community search
- **Graph-only retriever** — BFS subgraph traversal
- **Parent-child retriever** — child chunk retrieval → parent context fetch
- **Multi-query retriever** — LLM generates N query variants, merges via RRF
- **Microsoft GraphRAG retriever** — wraps MS GraphRAG local/global search
- **Adaptive retriever** — heuristic query classifier routes to best sub-retriever

### Added — Packaging
- **`pyproject.toml`** with optional dependency groups: `pdf`, `office`, `ocr`, `ocr-local`, `openai`, `azure`, `anthropic`, `bedrock`, `vertex-ai`, `cohere`, `ollama`, `sentence-transformers`, `neo4j`, `networkx`, `memgraph`, `arangodb`, `microsoft-graphrag`, `chroma`, `qdrant`, `pinecone`, `faiss`, `weaviate`, `milvus`, `pgvector`, `azure-search`, `nlp`, `default`, `all`

### Changed
- `hybrid_rag/main.py` — `build_pipeline()` now emits `DeprecationWarning` and delegates to `RAGLibrary`
- All models split into `cognity-ai/models/document.py`, `models/knowledge.py`, `models/retrieval.py`
- All configs split into `cognity-ai/config/base.py` (`LibraryConfig`) + `config/providers.py` (all provider configs)
- `NLPProcessor` split: extraction → `extractors/nlp.py`, chunking → `chunkers/sentence.py`, page index → `page_index/regex_index.py`
- `GeminiExtractor` split: embeddings → `embedders/gemini.py`, generation → `generators/gemini.py`, augmentation → `extractors/hybrid.py`, OCR → `ocr/gemini_vision.py`
- `GraphManager` → `stores/graph/neo4j.py` (implements `BaseGraphStore`)
- `VectorManager` → `stores/vector/chroma.py` (implements `BaseVectorStore`)
- `IngestionPipeline` → `pipeline/ingestion.py` (plugin-aware; uses ABCs instead of concrete classes)
- `HybridRetriever` → `retrievers/hybrid_graph.py` (implements `BaseRetriever`)
- `KnowledgeUpdater` → `pipeline/knowledge_updater.py` (uses `BaseGraphStore`)

### Deprecated
- `hybrid_rag.main.build_pipeline()` — replaced by `RAGLibrary`; will be removed in a future version
- Direct imports from `hybrid_rag.*` — use `cognity-ai.*` instead

---

## [1.0.0] - 2026-03-13

### Added
- **NLP-first extraction pipeline** — spaCy NER, SVO triples, dependency parsing, coreference resolution, noun phrase extraction
- **Gemini augmentation layer** — LLM fills semantic gaps NLP can't catch
- **4-channel hybrid retrieval** — Graph Local, Vector Semantic, Community Global, Graph→Vector Bridge
- **Reciprocal Rank Fusion (RRF)** with configurable channel weights
- **Microsoft GraphRAG patterns** — Leiden community detection, hierarchical summarization
- **Incremental ingestion** — SHA-256 hash-based change detection, stale data cleanup
- **Page/section indexing** — structural awareness with persistent page index
- **Knowledge lifecycle management** — confirm/deprecate sources, confidence propagation
- **Conflict detection** — surfaces contradictory triples across sources
- **Low-confidence pruning** — removes weak triples below threshold
- **Graph↔Vector bridge** — `ChunkRef` nodes with `MENTIONED_IN` edges connect graph entities to vector chunks
- **Dual ChromaDB collections** — separate chunk and community summary stores
- **Batch embeddings** — 100 texts per API call for cost efficiency
- **Built-in rate limiting** — configurable RPM safety on all Gemini calls
- **Health reporting** — entity/relation/doc/community counts with average confidence
- **Garbage collection** — `sync()` removes orphaned documents
