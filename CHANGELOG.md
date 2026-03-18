# Changelog

All notable changes to HybridGraphRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-03-19

### Added — Core Library
- **`raglib` package** — complete modular rewrite of `hybrid_rag` into a provider-agnostic library
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
- All models split into `raglib/models/document.py`, `models/knowledge.py`, `models/retrieval.py`
- All configs split into `raglib/config/base.py` (`LibraryConfig`) + `config/providers.py` (all provider configs)
- `NLPProcessor` split: extraction → `extractors/nlp.py`, chunking → `chunkers/sentence.py`, page index → `page_index/regex_index.py`
- `GeminiExtractor` split: embeddings → `embedders/gemini.py`, generation → `generators/gemini.py`, augmentation → `extractors/hybrid.py`, OCR → `ocr/gemini_vision.py`
- `GraphManager` → `stores/graph/neo4j.py` (implements `BaseGraphStore`)
- `VectorManager` → `stores/vector/chroma.py` (implements `BaseVectorStore`)
- `IngestionPipeline` → `pipeline/ingestion.py` (plugin-aware; uses ABCs instead of concrete classes)
- `HybridRetriever` → `retrievers/hybrid_graph.py` (implements `BaseRetriever`)
- `KnowledgeUpdater` → `pipeline/knowledge_updater.py` (uses `BaseGraphStore`)

### Deprecated
- `hybrid_rag.main.build_pipeline()` — replaced by `RAGLibrary`; will be removed in a future version
- Direct imports from `hybrid_rag.*` — use `raglib.*` instead

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
