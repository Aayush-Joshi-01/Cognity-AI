# Architecture & Design — raglib

Deep dive into the design decisions, algorithms, and data flow patterns powering the raglib modular RAG library.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Package Structure](#2-package-structure)
3. [Plugin & Registry Architecture](#3-plugin--registry-architecture)
4. [The Ingestion Pipeline](#4-the-ingestion-pipeline)
5. [Extraction Pipeline (NLP-First + LLM Augmentation)](#5-extraction-pipeline-nlp-first--llm-augmentation)
6. [Chunking Strategies](#6-chunking-strategies)
7. [Page Index Subsystem](#7-page-index-subsystem)
8. [OCR Subsystem](#8-ocr-subsystem)
9. [Graph Data Model](#9-graph-data-model)
10. [Microsoft GraphRAG: Community Detection & Summarization](#10-microsoft-graphrag-community-detection--summarization)
11. [The Graph↔Vector Bridge](#11-the-graphvector-bridge)
12. [4-Channel Hybrid Retrieval](#12-4-channel-hybrid-retrieval)
13. [Reciprocal Rank Fusion (RRF)](#13-reciprocal-rank-fusion-rrf)
14. [RAG Methodology Selection & Auto-Fallback](#14-rag-methodology-selection--auto-fallback)
15. [Incremental Ingestion](#15-incremental-ingestion)
16. [Knowledge Lifecycle & Confidence Model](#16-knowledge-lifecycle--confidence-model)
17. [Cost Optimization Strategy](#17-cost-optimization-strategy)
18. [Failure Modes & Graceful Degradation](#18-failure-modes--graceful-degradation)

---

## 1. Design Philosophy

The library is built on three core principles:

**Provider-agnostic by design.** Every component — LLM generator, embedder, vector store, graph store, chunker, loader, OCR provider — is expressed as an abstract base class (`Base*`) with string-keyed concrete implementations registered in a central `PluginRegistry`. Swapping providers is a single config string change: `embedder="openai"` instead of `embedder="gemini"`. No provider-specific code leaks into the pipeline layer. The `ComponentFactory` (`factory.py`) reads `LibraryConfig` string keys and instantiates the right classes using lazy imports inside `try/except` blocks, so missing optional dependencies do not crash import of the library itself.

**Extract locally, augment remotely.** spaCy handles deterministic NLP tasks — named entity recognition, dependency parsing, subject-verb-object extraction, and lightweight coreference resolution — at zero API cost. LLMs only handle what requires semantic reasoning: causal chains, implicit associations, part-of hierarchies not syntactically marked, and complex layout understanding. This principle applies to both knowledge extraction (spaCy first, LLM fills gaps) and OCR (Tesseract for simple printed text, LLM vision APIs for complex layouts, tables, and handwriting). The cost difference is substantial: a 1,000-chunk corpus runs NLP extraction for free, then makes selective LLM calls only where NLP found entities worth augmenting.

**Multiple retrieval channels beat any single one.** No single retrieval method dominates across all query types. Factual lookups ("Who founded OpenAI?") favor graph traversal. Semantic similarity queries favor vector search. Broad thematic questions ("What are the major AI safety themes?") favor community summaries. Entity-specific context lookups benefit from the Graph→Vector bridge. The system runs all active channels in parallel and fuses their ranked results with Reciprocal Rank Fusion (RRF), letting the overlap signal determine relevance.

---

## 2. Package Structure

```
raglib/
├── library.py            RAGLibrary — unified public API facade
├── factory.py            build_components() + build_retriever() — wires everything from config
├── registry.py           PluginRegistry — class-level dicts for all component types
│
├── models/
│   ├── document.py       Document, ImageRef
│   ├── knowledge.py      Entity, Relation, ExtractionResult, SourceStatus
│   └── retrieval.py      SemanticChunk, PageInfo, CommunityInfo, DocumentMeta, RetrievalResult
│
├── config/
│   ├── base.py           LibraryConfig — all component selections + nested provider configs
│   └── providers.py      Neo4jConfig, GeminiConfig, OpenAIConfig, AnthropicConfig, ...
│
├── loaders/
│   ├── base.py           BaseLoader ABC
│   ├── factory.py        LoaderFactory — extension → loader dispatch
│   ├── pdf.py            PdfLoader (pdfplumber primary, pypdf/pdfminer fallback)
│   ├── docx.py           DocxLoader (python-docx, embedded image extraction)
│   ├── excel.py          ExcelLoader (openpyxl, per-sheet Documents)
│   ├── pptx.py           PptxLoader (python-pptx, per-slide text + images)
│   ├── html.py           HtmlLoader (BeautifulSoup)
│   ├── csv.py            CsvLoader
│   ├── json_loader.py    JsonLoader
│   ├── text.py           TextLoader (.txt, .md, .rst, .yaml)
│   ├── image.py          ImageLoader (routes to OCR subsystem)
│   └── pdf_utils.py      Shared PDF page extraction helpers
│
├── ocr/
│   ├── base.py           BaseOCR ABC
│   ├── factory.py        OCRFactory.create() + create_with_fallback()
│   ├── gemini_vision.py  GeminiVisionOCR (gemini-2.0-flash multimodal)
│   ├── openai_vision.py  OpenAIVisionOCR (gpt-4o vision)
│   ├── anthropic_vision.py AnthropicVisionOCR (claude-sonnet-4-6 vision)
│   ├── azure_vision.py   AzureVisionOCR (Azure GPT-4o)
│   ├── bedrock_vision.py BedrockVisionOCR (AWS Bedrock Claude)
│   └── tesseract.py      TesseractOCR (pytesseract, offline fallback)
│
├── chunkers/
│   ├── base.py           BaseChunker ABC
│   ├── sentence.py       SentenceChunker (spaCy sentences, N-sentence windows)
│   ├── fixed.py          FixedChunker (word-count windows)
│   ├── recursive.py      RecursiveChunker (paragraph → sentence → word splits)
│   ├── semantic.py       SemanticChunker (embedding cosine similarity grouping)
│   ├── parent_child.py   ParentChildChunker (large parent + small child)
│   └── hybrid.py         HybridChunker (sentence primary + optional semantic regroup)
│
├── page_index/
│   ├── base.py           BasePageIndex ABC
│   ├── json_store.py     JsonPageStore — persistence layer for all index types
│   ├── regex_index.py    RegexPageIndex (form feeds, headings, page markers)
│   ├── structural_index.py StructuralPageIndex (reads Document.page_map from loader)
│   └── hybrid_index.py   HybridPageIndex (structural primary, regex fallback)
│
├── extractors/
│   ├── base.py           BaseExtractor ABC
│   ├── nlp.py            NLPExtractor (pure spaCy: NER + deps + SVO + coref + noun phrases)
│   ├── llm.py            LLMExtractor (LLM-only via any BaseGenerator)
│   └── hybrid.py         HybridExtractor (NLP first, LLM fills gaps)
│
├── embedders/
│   ├── base.py           BaseEmbedder ABC
│   ├── gemini.py         GeminiEmbedder
│   ├── vertex_ai.py      VertexAIEmbedder
│   ├── openai.py         OpenAIEmbedder
│   ├── azure_openai.py   AzureOpenAIEmbedder
│   ├── bedrock.py        BedrockEmbedder
│   ├── cohere.py         CohereEmbedder
│   ├── sentence_transformers.py  SentenceTransformerEmbedder (local, offline)
│   └── ollama.py         OllamaEmbedder (local Ollama server)
│
├── generators/
│   ├── base.py           BaseGenerator ABC
│   ├── gemini.py         GeminiGenerator
│   ├── vertex_ai.py      VertexAIGenerator
│   ├── openai.py         OpenAIGenerator
│   ├── azure_openai.py   AzureOpenAIGenerator
│   ├── anthropic.py      AnthropicGenerator
│   ├── bedrock.py        BedrockGenerator
│   ├── cohere.py         CohereGenerator
│   └── ollama.py         OllamaGenerator
│
├── stores/
│   ├── vector/
│   │   ├── base.py       BaseVectorStore ABC
│   │   ├── chroma.py     ChromaStore (default, local or remote)
│   │   ├── qdrant.py     QdrantStore
│   │   ├── pinecone.py   PineconeStore
│   │   ├── faiss.py      FAISSStore (local, no metadata persistence)
│   │   ├── weaviate.py   WeaviateStore
│   │   ├── milvus.py     MilvusStore
│   │   ├── pgvector.py   PgVectorStore (PostgreSQL + pgvector)
│   │   └── azure_search.py AzureSearchStore
│   └── graph/
│       ├── base.py       BaseGraphStore ABC
│       ├── neo4j.py      Neo4jStore (primary; GDS for community detection)
│       ├── microsoft_graphrag.py  MicrosoftGraphRAGStore (wraps official library)
│       ├── memgraph.py   MemgraphStore (MAGE community algorithms)
│       ├── arangodb.py   ArangoDBStore
│       └── networkx.py   NetworkXStore (in-memory, zero infrastructure)
│
├── retrievers/
│   ├── base.py           BaseRetriever ABC
│   ├── hybrid_graph.py   HybridGraphRetriever (4-channel: graph + vector + community + bridge)
│   ├── naive.py          NaiveRetriever (pure vector similarity)
│   ├── vector_only.py    VectorOnlyRetriever (vector + community, no graph)
│   ├── graph_only.py     GraphOnlyRetriever (graph traversal only)
│   ├── parent_child.py   ParentChildRetriever (child retrieval → fetch parent)
│   ├── multi_query.py    MultiQueryRetriever (LLM generates N query variants)
│   ├── microsoft_graphrag.py  MicrosoftGraphRAGRetriever
│   └── adaptive.py       AdaptiveRetriever (heuristic query classifier)
│
├── pipeline/
│   ├── ingestion.py      IngestionPipeline — 9-step orchestration
│   └── knowledge_updater.py  KnowledgeUpdater — lifecycle + conflict detection + pruning
│
└── utils/
    ├── hash.py           HashStore (SHA-256 change detection), content_hash()
    ├── rrf.py            reciprocal_rank_fusion()
    └── token_counter.py  Word-count token estimation
```

`RAGLibrary` (`library.py`) is the single entry point users interact with. It constructs a `LibraryConfig`, calls `build_components()` to wire all concrete implementations, instantiates `IngestionPipeline` and `KnowledgeUpdater`, and builds the default retriever. Advanced users can bypass `RAGLibrary` and compose `IngestionPipeline` directly.

---

## 3. Plugin & Registry Architecture

Every component type follows the same four-layer pattern:

**Layer 1 — Abstract Base Class.** Defines the interface contract that all implementations must honor. For example, `BaseEmbedder` declares `embed(text) -> list[float]` and `embed_batch(texts) -> list[list[float]]`. No pipeline code imports a concrete class; it always works through the ABC.

**Layer 2 — Concrete Implementations.** Provider-specific classes (e.g., `GeminiEmbedder`, `OpenAIEmbedder`, `SentenceTransformerEmbedder`) implement the ABC. Each lives in its own module so its import only occurs when that provider is actually requested.

**Layer 3 — PluginRegistry.** Class-level dictionaries indexed by string key. Built-in implementations are pre-registered. Users register custom implementations through `RAGLibrary.register_loader()`, `register_chunker()`, etc., which delegate directly to `PluginRegistry`:

```python
# Built-in registration (in each module's __init__)
PluginRegistry.register_embedder("gemini", GeminiEmbedder)
PluginRegistry.register_embedder("openai", OpenAIEmbedder)

# User-defined plugin registration
rag.register_loader(".myext", MyCustomLoader)
rag.register_embedder("my_embedder", MyEmbedder)
```

**Layer 4 — ComponentFactory.** `factory.py`'s `build_components(cfg: LibraryConfig)` reads the string keys from `LibraryConfig` and instantiates the right classes. It uses explicit lazy imports (`from raglib.embedders.gemini import GeminiEmbedder` inside the `if key == "gemini":` branch) so missing optional dependencies raise `ImportError` only at instantiation time, never at module import time.

```
LibraryConfig(embedder="openai", vector_store="qdrant", graph_store="neo4j", llm="anthropic", ...)
        │
        ▼
build_components(cfg)
        │
        ├── _build_nlp_model(cfg)          → spacy.Language | None
        ├── _build_ocr(cfg)                → BaseOCR (with tesseract fallback)
        ├── _build_embedder("openai", cfg) → OpenAIEmbedder(cfg.openai)
        ├── _build_generator("anthropic")  → AnthropicGenerator(cfg.anthropic)
        ├── _build_graph_store(cfg)        → Neo4jStore(cfg.neo4j, cfg.graphrag)
        ├── _build_vector_store(cfg)       → QdrantStore(cfg.qdrant)
        ├── _build_extractor(cfg, nlp, gen)→ HybridExtractor(NLPExtractor, LLMExtractor)
        ├── _build_page_index(cfg)         → HybridPageIndex(store_path=...)
        ├── _build_chunker(cfg, nlp, emb)  → SentenceChunker(nlp_model, sentences=5)
        └── HashStore(path=cfg.ingestion.hash_store_path)
        │
        ▼
RAGLibrary holds all components, builds IngestionPipeline + retriever
```

The factory also applies auto-fallback logic before returning components (see Section 14 for the full fallback table).

---

## 4. The Ingestion Pipeline

`IngestionPipeline.ingest()` executes nine sequential steps. Each step catches its own exceptions so a failure in one step (e.g., embedding API timeout) does not abort the pipeline — it degrades gracefully.

```
rag.ingest("report.pdf")
    │
    ▼ LoaderFactory.load(path, ocr=ocr_provider)
    │   ├── Extension → Loader mapping (.pdf → PdfLoader, .docx → DocxLoader, ...)
    │   ├── PdfLoader: pdfplumber primary (page text + boundaries)
    │   │             pypdf / pdfminer fallback if pdfplumber fails
    │   ├── DocxLoader: python-docx, extracts embedded images as ImageRef objects
    │   ├── ExcelLoader: per-sheet Documents (sheet name becomes doc_id suffix)
    │   ├── ImageLoader: routes bytes/path to the OCR subsystem
    │   └── Returns list[Document] (usually one; Excel/PPTX may produce several)
    │
    ▼ Document(text, page_map, image_refs, metadata, ...)
    │   image_refs: list[ImageRef(image_id, char_offset, image_bytes, ocr_text)]
    │   page_map:   list[{page_num, start_char, end_char, heading}] from loader
    │
    ▼ Step 1: HashStore.is_unchanged(doc_id, text)?
    │   └── YES → return {"status": "skipped", "reason": "unchanged"}
    │
    ▼ [If doc previously existed] Clear stale data
    │   ├── graph_store.remove_doc_subgraph(doc_id)   — relations, orphan entities, ChunkRefs
    │   ├── vector_store.delete_by_doc_id(doc_id)
    │   └── page_index.remove(doc_id)
    │
    ▼ Step 2: Page/section detection
    │   HybridPageIndex.detect_pages(text, doc_id, loader_metadata)
    │   page_index.persist()   → page_index.json updated
    │
    ▼ Step 3: Chunking
    │   chunker.chunk(text, doc_id, pages) → list[SemanticChunk]
    │   Each chunk carries: chunk_id, doc_id, text, index, page_info, token_estimate
    │
    ▼ Step 4: Knowledge extraction (per chunk)
    │   for chunk in chunks:
    │       result = extractor.extract(chunk.text, source_id=doc_id)
    │       → ExtractionResult(entities: list[Entity], relations: list[Relation])
    │       chunk.entity_names = [e.name for e in result.entities]
    │
    ▼ Step 5: Cross-chunk entity/relation deduplication
    │   entity_map keyed on name.lower().strip():
    │     - Higher confidence wins for entity_type and confidence
    │     - mentions accumulate (frequency signal across chunks)
    │     - extraction_method becomes "merged" when both NLP and LLM contributed
    │   rel_map keyed on "source|TYPE|target":
    │     - weight accumulates (frequency signal)
    │     - Higher confidence wins, merged weight carried forward
    │
    ▼ Step 6: Batch embedding
    │   embedder.embed_batch([chunk.text for chunk in chunks])
    │   Batched per provider defaults (Gemini: 100/call)
    │   chunk.embedding set for each chunk
    │
    ▼ Step 7: Graph upsert
    │   for entity in unique_entities: graph_store.upsert_entity(entity)
    │   for relation in unique_relations: graph_store.upsert_relation(relation)
    │   for chunk in chunks:
    │       graph_store.link_chunk_to_entities(chunk_id, doc_id, chunk.entity_names)
    │   graph_store.upsert_doc_meta(doc_id, hash, source_name, status, stats)
    │
    ▼ Step 8: Vector upsert
    │   vector_store.upsert_chunks([c for c in chunks if c.embedding is not None])
    │
    ▼ Step 9: Hash update
    │   hash_store.set(doc_id, SHA-256(text))
    │
    ▼ return {
        "doc_id": doc_id, "status": "ingested",
        "pages": N, "chunks": N, "entities": N, "relations": N
      }
```

**Why structural + regex hybrid page detection?** PDF loaders return exact page boundaries from the PDF spec (via pdfplumber's `page_map`). Text files, Markdown, and HTML have no such metadata. The `HybridPageIndex` checks whether `loader_metadata` was provided: if yes, it delegates to `StructuralPageIndex` (most accurate, uses actual page numbers and headings from the loader); otherwise, it falls back to `RegexPageIndex` (pattern matching on form feeds, `---`, `# headings`, `Page N`). Both are backed by `JsonPageStore` at the same file path, so queries always find the data.

**Why cross-chunk deduplication?** The same entity ("Sam Altman", "sam altman", "Altman") can appear across dozens of chunks. Without deduplication, the graph would receive dozens of `upsert_entity` calls for what is conceptually one node. The pipeline normalizes to `name.lower().strip()` as the deduplication key, accumulates `mentions` (cross-chunk frequency), and carries forward the highest-confidence version of conflicting attributes. This means the graph's `mentions` count reflects the true document-wide frequency of an entity, not per-chunk occurrence.

**Embedded image handling in DOCX, PPTX, and PDF.** Loaders extract embedded images as `ImageRef` objects with a `char_offset` (the character position in `Document.text` where the image was encountered) and `image_bytes`. The OCR subsystem processes each `ImageRef`, and the resulting `ocr_text` is stored back on the `ImageRef`. During chunking, chunks near that `char_offset` incorporate the OCR text into their context. This preserves semantic continuity between surrounding prose and image content (charts, tables, diagrams).

---

## 5. Extraction Pipeline (NLP-First + LLM Augmentation)

Extraction is handled by three concrete `BaseExtractor` implementations selected via `LibraryConfig.extraction`.

### NLPExtractor (`extractors/nlp.py`)

Pure spaCy, zero API cost. Runs four techniques on every chunk:

**Named Entity Recognition (NER)**
spaCy's transformer-based NER identifies entities and maps them to an ontology:

```
PERSON → Person        ORG → Organization      GPE/LOC → Location
PRODUCT → Product      EVENT → Event            NORP → Group
WORK_OF_ART → CreativeWork    DATE/TIME → Temporal
```

Entities are normalized to title case. Pure numeric or temporal entities shorter than 4 characters are filtered (noise for knowledge graphs).

**Dependency-Based Relation Extraction**
For each sentence containing 2+ named entities, the parser walks the dependency tree between entity root tokens up to 4 hops, mapping dependency arcs to semantic relation types:

```
nsubj → SUBJECT_OF     dobj → ACTS_ON       pobj → RELATES_TO
attr → IS_A            appos → ALSO_KNOWN_AS    compound → PART_OF
agent → ACTED_BY       poss → BELONGS_TO
```

When both entities share a verbal head (siblings in the parse tree), the verb lemma becomes the relation type directly.

**Subject-Verb-Object (SVO) Triple Extraction**
A separate pass identifies every `VERB` token, collects its `nsubj`/`nsubjpass` children as subjects and `dobj`/`attr`/`pobj` children (including through prepositional phrases) as objects. Each subject-object pair yields a triple with the verb lemma as the relation. Compound expansion collects `compound`, `amod`, and `flat` children so "Sam Altman" isn't split into two separate entities. Common verbs are normalized to semantic relation types: `FOUND → FOUNDED_BY`, `DEVELOP → DEVELOPED`, `BASE → BASED_IN`.

**Pronominal Coreference Resolution**
A lightweight heuristic resolves pronouns to their most likely antecedent without a full coreference model:

- `he/she/him/her/his` → nearest preceding `PERSON` entity
- `it/its` → nearest preceding `ORG/PRODUCT/GPE` entity
- `they/them/their` → nearest preceding entity of any type

This resolves approximately 70–80% of pronominal references at zero cost. Relations extracted with pronouns as source/target are rewritten to the resolved entity name.

**Noun Phrase Entity Extraction**
spaCy's noun chunker catches compound concepts that NER misses. Phrases are filtered to keep only those with adjective or noun modifiers (e.g., "Constitutional AI", "protein folding") and are assigned `entity_type = "Concept"` with lower confidence (0.6).

### LLMExtractor (`extractors/llm.py`)

Uses any `BaseGenerator` implementation. Sends the full chunk text to the LLM with a structured prompt requesting JSON output: `{"entities": [...], "relations": [...]}`. No prior NLP context is passed — the LLM operates on raw text. Suitable when spaCy is unavailable or when LLM-quality extraction is required for every chunk regardless of cost.

### HybridExtractor (`extractors/hybrid.py`)

The default. Combines both extractors in configurable modes:

| Mode | Behavior |
|------|----------|
| `augment` (default) | NLP runs first. LLM receives chunk text + already-extracted entities/relations. Its prompt explicitly instructs: find ADDITIONAL semantic relationships that NLP missed. Do NOT repeat what is already extracted. This targets what spaCy genuinely cannot do: causal chains, implicit associations, temporal sequences, part-of hierarchies not syntactically marked. |
| `nlp_only` | Skip the LLM step entirely. All extraction is free and local. |
| `llm_only` | Skip NLP and call the LLM directly. Falls back to NLP if no LLM is configured. |

After augmentation, entity and relation lists from both extractors are merged and deduplicated within the chunk before being returned.

### Default Confidence Values

| Source | Default Confidence |
|--------|--------------------|
| spaCy NER entities | 0.85 |
| Dependency-parsed relations | 0.75 |
| SVO triple relations | 0.70 |
| Noun phrase entities | 0.60 |
| LLM-augmented entities | 0.90 |
| LLM-augmented relations | 0.85 |

---

## 6. Chunking Strategies

All chunkers implement `BaseChunker.chunk(text, doc_id, pages) -> list[SemanticChunk]`. Each returned `SemanticChunk` carries `chunk_id`, `doc_id`, `text`, `index`, `page_info`, `token_estimate`, and (after embedding) `embedding` and `entity_names`.

**`sentence` (default)** — spaCy sentence boundary segmentation. Groups N consecutive sentences into one chunk with one-sentence overlap between adjacent chunks. Default: `chunk_sentences=5`, `overlap=1`. Produces no broken sentences, which improves both embedding quality and NER accuracy (NER runs on complete sentences). Best for: general-purpose corpora.

**`fixed`** — Word-count windows with configurable size and overlap. No NLP dependency. Best for: environments without spaCy, preprocessing speed priority, or fixed-size tokenizer requirements.

**`recursive`** — Splits hierarchically: paragraph boundaries first, then sentence boundaries, then word boundaries, stopping when chunk size is within target range. Mirrors the LangChain `RecursiveCharacterTextSplitter` pattern. Best for: mixed-format text (prose mixed with code or structured lists) where paragraph-level coherence matters more than exact sentence count.

**`semantic`** — Groups consecutive sentences by embedding cosine similarity. Sentences with similar embeddings are merged into one chunk; dissimilar sentence pairs become chunk boundaries. Requires the configured embedder (one embedding call per sentence during chunking). Best for: topic-coherent chunks where downstream retrieval precision is more important than chunking cost.

**`parent_child`** — Produces two sizes: large parent chunks (broad context) and small child chunks (precise retrieval units). Parent chunks are stored with `is_parent=True`; child chunks store `parent_chunk_id` pointing to their parent. The `ParentChildRetriever` retrieves children by similarity, then fetches their parents for generation context. Best for: long documents where retrieval precision on specific facts matters but generation requires surrounding context.

**`hybrid`** — Sentence chunking as primary strategy, with optional semantic regrouping of adjacent chunks that are highly similar. Balances chunking cost against semantic coherence. Best for: corpora where topic drift within a document is expected and sentence-boundary chunking alone produces chunks that straddle topic transitions.

---

## 7. Page Index Subsystem

The page index provides structural awareness — which page, section, or heading does a given character offset fall within? This metadata propagates through chunks to retrieval results, enabling page-filtered queries ("What is on page 5?") and section-aware retrieval.

`BasePageIndex` defines the interface:

```python
class BasePageIndex(ABC):
    def detect_pages(self, text, doc_id, loader_metadata=None) -> list[PageInfo]: ...
    def get_page_for_char(self, doc_id, char_offset) -> PageInfo | None: ...
    def store(self, doc_id, pages: list[PageInfo]): ...
    def get(self, doc_id) -> list[PageInfo]: ...
    def remove(self, doc_id): ...
    def persist(self): ...
```

`PageInfo` carries: `page_num`, `section`, `start_char`, `end_char`, `heading`.

### Three Implementations

**RegexPageIndex** — Pure pattern matching on the raw text. Detects:
- Form feed characters (`\f`) — universal page break marker
- Horizontal rules (`---`, `===`) — Markdown and text section separators
- Explicit page markers (`Page N`, `Page N of M`)
- Markdown headings (`# Title`, `## Section`)
- Numbered sections (`1. Introduction`, `2.1 Background`)

No loader metadata required. Works on any text format. Used as fallback when structured metadata is unavailable.

**StructuralPageIndex** — Reads `Document.page_map` directly from the loader. `PdfLoader` populates `page_map` from pdfplumber's page boundaries (exact character offsets per PDF page). `ExcelLoader` uses sheet names and row offsets. `PptxLoader` uses slide numbers. Most accurate for structured formats because it reflects actual document structure rather than heuristic pattern matching.

**HybridPageIndex (default)** — Routes to `StructuralPageIndex` when `loader_metadata` is provided and non-empty; routes to `RegexPageIndex` otherwise. Both delegate to the same `JsonPageStore` backing file path, so `get_page_for_char()` always finds data regardless of which strategy produced it. All three implementations persist their results via `JsonPageStore` (a JSON file at `cfg.ingestion.page_index_path`). On incremental re-ingestion, stale entries are removed before new pages are detected.

---

## 8. OCR Subsystem

The OCR subsystem converts images to text. It is invoked in two contexts: (1) directly by `ImageLoader` when the input file is an image, and (2) during loader processing of DOCX/PPTX/PDF files that contain embedded images as `ImageRef` objects.

All providers implement `BaseOCR.ocr(image: bytes | str) -> str`.

```
ImageLoader receives .jpg / .png / .bmp / .tiff / etc.
    │
    ▼ BaseOCR.ocr(image_path_or_bytes) → str
    │
    ├── GeminiVisionOCR     (DEFAULT — gemini-2.0-flash multimodal)
    │   └── Handles complex layouts, tables, handwriting, mixed text/graphics
    │
    ├── OpenAIVisionOCR
    │   └── gpt-4o vision — high accuracy, structured document content
    │
    ├── AnthropicVisionOCR
    │   └── claude-sonnet-4-6 vision — strong document understanding
    │
    ├── AzureVisionOCR
    │   └── Azure-hosted GPT-4o — enterprise Azure environments
    │
    ├── BedrockVisionOCR
    │   └── AWS Bedrock Claude — AWS-locked deployments
    │
    └── TesseractOCR        (FALLBACK — pytesseract, offline)
        └── Simple printed text, no API cost, lower accuracy on complex layouts
```

`OCRFactory.create_with_fallback(providers, config)` tries each provider in the given order, returning the first that instantiates successfully. The default fallback chain used by `_build_ocr()` in the factory is: `gemini_vision → tesseract`. If `gemini_vision` fails to initialize (missing API key, network issue), Tesseract is attempted.

All LLM-based OCR providers support `supports_multimodal = True`, enabling richer prompting that instructs the model to preserve table structure, extract caption text, and handle multi-column layouts. Tesseract does not support multimodal prompting.

**Embedded image workflow in DOCX/PPTX/PDF:**
1. Loader extracts embedded images as bytes into `Document.image_refs` — each `ImageRef` carries `char_offset` (its position in the parent document text), `image_bytes`, `mime_type`, and `page_num`.
2. `ImageLoader` (or the parent loader) calls the configured OCR provider for each `ImageRef`.
3. The resulting text is stored in `ImageRef.ocr_text`.
4. During chunking, chunks whose character span includes the `char_offset` of an `ImageRef` are augmented with the OCR text, preserving semantic continuity between surrounding prose and image content.

---

## 9. Graph Data Model

The graph data model is implemented in `stores/graph/neo4j.py` (primary) and mirrored across other `BaseGraphStore` implementations. The same node/edge schema is expressed in each backend's native query language.

### Node Types

**Entity** — the primary knowledge unit.

```
(:Entity {
    name: "Sam Altman",              // Canonical name (title case, normalized)
    entity_type: "Person",           // Ontology type from extraction
    description: "CEO of OpenAI",   // Longest description wins on upsert
    confidence: 0.9,                 // 0.0–1.0; highest wins on upsert
    mentions: 5,                     // Cross-chunk frequency; accumulates on upsert
    source_id: "doc_001",           // Origin document
    extraction_method: "merged",     // "nlp" | "llm" | "merged"
    created_at: datetime,
    updated_at: datetime
})
```

Entities also receive a **dynamic secondary label** matching their type (`:Person`, `:Organization`, `:Location`, etc.) for efficient type-filtered Cypher queries without scanning all `Entity` nodes.

**DocumentMeta** — source tracking for lifecycle management.

```
(:DocumentMeta {
    doc_id, source_name, content_hash, status,
    chunk_count, entity_count, relation_count,
    ingested_at, updated_at
})
```

**Community** — GraphRAG community summaries (stored after `build_communities()`).

```
(:Community {
    community_id, level, title, summary, rank, entity_count
})
```

**ChunkRef** — lightweight bridge node linking the graph to the vector store.

```
(:ChunkRef { chunk_id, doc_id })
```

### Edge Types

**Dynamic relation edges** between `Entity` nodes use the extracted relation type as the Cypher relationship label: `FOUNDED_BY`, `DEVELOPED`, `HEADQUARTERED_IN`, `RAISED`, etc. Each carries:

```
{ confidence, weight, description, source_id, extraction_method, created_at }
```

**Structural edges:**
- `(:Entity)-[:MENTIONED_IN]->(:ChunkRef)` — the graph↔vector bridge
- `(:Entity)-[:BELONGS_TO_COMMUNITY]->(:Community)` — community membership

### Upsert Semantics

Entities use `MERGE` on `name`. On match:
- Higher `confidence` wins for `entity_type` and `confidence`
- Longer `description` wins (more informative)
- `mentions` accumulates (sum across upserts)
- `extraction_method` becomes `"merged"` if both NLP and LLM contributed at any point

Relations use `MERGE` on `(source_entity)-[TYPE]->(target_entity)`. On match:
- Higher `confidence` wins
- `weight` accumulates (frequency signal across documents)
- Longer `description` wins

Re-ingesting documents enriches existing knowledge rather than duplicating it.

---

## 10. Microsoft GraphRAG: Community Detection & Summarization

The system implements the core pattern from Microsoft's GraphRAG (Edge et al., 2024): detect communities of densely connected entities, summarize each community with an LLM, and use those summaries for global search over broad thematic queries.

### Community Detection

Community detection is invoked via `IngestionPipeline.build_communities()` after all documents are ingested. It calls `graph_store.detect_communities()`, whose implementation varies by backend:

| Graph Store | Community Algorithm |
|-------------|---------------------|
| Neo4j (+ GDS) | Leiden (preferred) → Louvain (fallback if Leiden plugin missing) |
| Neo4j (no GDS) | Skip community detection |
| Memgraph | MAGE `community_detection` module |
| NetworkX | `greedy_modularity_communities` (scikit-network) |
| MicrosoftGraphRAGStore | Delegates to the official `microsoft/graphrag` library |
| ArangoDB | Pregel-based community detection |

**Leiden process (Neo4j + GDS):**
1. Project all `Entity` nodes and their relationships into a GDS in-memory graph
2. Run Leiden with configurable resolution (higher resolution = more granular communities)
3. Write community assignments back to entity nodes as `community_level_0`
4. Intermediate levels stored in `community_levels` for hierarchical access
5. Drop the GDS projection

**Why Leiden over Louvain:** Leiden guarantees well-connected communities (no disconnected subclusters), runs faster on large graphs, and produces more stable results across runs. Louvain is retained as a fallback for environments where the Leiden GDS plugin is unavailable.

### Community Summarization

For each detected community with 2 or more members:
1. Retrieve all intra-community entities and their relationships from the graph
2. Format entity names and relationship descriptions into a structured prompt
3. LLM generates a `title` (2–5 words) and `summary` (2–3 sentences) as JSON
4. The `title + summary` string is embedded via the configured embedder
5. The `CommunityInfo` object is stored as a `Community` node in the graph store AND upserted into the vector store's community collection

### Global Search

Two parallel community retrieval paths run during hybrid retrieval:
1. **Graph-side:** Communities ranked by `rank` (proportion of total graph entities they contain — larger communities rank higher as they represent major themes)
2. **Vector-side:** Cosine similarity between the query embedding and community summary embeddings in the vector store

---

## 11. The Graph↔Vector Bridge

The bridge connects the knowledge graph with the vector store bidirectionally, solving a fundamental tension: graph retrieval returns structured relationship triples but lacks rich textual context; vector retrieval returns relevant text passages but cannot follow relationships.

**The bridge mechanism:** During ingestion (Step 7), after each chunk's entities are extracted, `graph_store.link_chunk_to_entities(chunk_id, doc_id, entity_names)` creates `ChunkRef` nodes and `MENTIONED_IN` edges:

```
(:Entity {name: "Sam Altman"})-[:MENTIONED_IN]->(:ChunkRef {chunk_id: "doc_001__chunk_3"})
(:Entity {name: "OpenAI"})-[:MENTIONED_IN]->(:ChunkRef {chunk_id: "doc_001__chunk_3"})
```

**During retrieval (Channel 4 in HybridGraphRetriever):**
1. NLP extracts seed entity names from the query text
2. Graph query: `graph_store.get_chunks_for_entities(entity_names)` — finds all `ChunkRef` nodes linked to those entities via `MENTIONED_IN` edges
3. Vector fetch: retrieve those specific chunks from the vector store by ID
4. Result: exact text passages that are guaranteed to contain the graph-identified entities

This gives the retriever high-signal textual context without relying on embedding similarity alone. A chunk retrieved via the bridge is entity-relevant by construction, not by approximation.

---

## 12. 4-Channel Hybrid Retrieval

`HybridGraphRetriever` (`retrievers/hybrid_graph.py`) runs up to four retrieval channels in parallel and fuses results with RRF.

### Channel 1: Graph Local Search

**Trigger:** Active when NLP extracts seed entities from the query.

**Method:** BFS subgraph expansion starting from fuzzy-matched entity nodes. In Neo4j, this uses APOC's `subgraphAll` with configurable hop depth (default 2). Falls back to manual Cypher BFS if APOC is not installed. Returns relationship triples with descriptions, confidence scores, and source attribution. Additionally, for the top 3 seed entities, `retrieve_entity_context()` fetches all direct relationships (both incoming and outgoing) for a complete local neighborhood view.

**Strengths:** Multi-hop reasoning ("Who funded the company that built GPT-4?"), relationship-aware, high precision for factual queries.
**Weaknesses:** Quality depends on entity extraction quality from the query; misses queries that don't name entities explicitly.

### Channel 2: Vector Semantic Search

**Trigger:** Always active.

**Method:** Embed the query with task type `retrieval_query` (where the embedder supports task types, as Gemini does), then cosine similarity search against the chunk collection in the vector store. Returns ranked `SemanticChunk` texts.

**Strengths:** Handles semantic paraphrasing, synonym matching, works even when entities are not explicitly named.
**Weaknesses:** No structural relationship awareness; can surface topically similar but factually irrelevant chunks.

### Channel 3: Community Global Search

**Trigger:** Active by default when communities have been built.

**Method:** Dual path — graph-ranked community summaries (by `rank` score, proportional to entity count) plus vector-similarity search over community summary embeddings.

**Strengths:** Answers broad thematic queries that no individual entity or chunk can address. Provides high-level context for synthesis questions.
**Weaknesses:** Lower specificity; most useful as supplementary context rather than primary answer source. Disabled for FAISS (no metadata storage for per-community lookup).

### Channel 4: Graph→Vector Bridge

**Trigger:** Active when seed entities are found (when graph store is available).

**Method:** Graph query finds `ChunkRef` nodes linked to seed entities via `MENTIONED_IN`, then fetches those specific chunks from the vector store by ID.

**Strengths:** High-signal context guaranteed to contain relevant entities. Combines graph precision with vector-stored text richness.
**Weaknesses:** Limited to chunks that contain explicitly extracted entity names; misses chunks that describe entities indirectly.

### Additional Retriever Methods

Beyond `hybrid_graph`, the following retriever implementations are available:

| Method | Description |
|--------|-------------|
| `naive` | Pure vector similarity. Simple, fast, no graph dependency. |
| `vector_only` | Vector + community search, no graph traversal. Good intermediate option. |
| `graph_only` | Graph traversal only, no vector search. For relationship-heavy queries. |
| `parent_child` | Retrieves child chunks by similarity, then fetches parent chunks for generation. |
| `multi_query` | LLM generates N query variants; retrieves for each; fuses results. Improves recall on ambiguous queries. |
| `microsoft_graphrag` | Delegates to the official `microsoft/graphrag` library for full GraphRAG-style retrieval. |
| `adaptive` | Heuristic query classifier selects `hybrid_graph` vs `naive` based on query characteristics (entity presence, question type). |

---

## 13. Reciprocal Rank Fusion (RRF)

All retrieval channels produce independently ranked result lists. `reciprocal_rank_fusion()` (`utils/rrf.py`) merges them into a single ranked list.

**Formula:**

```
score(d) = Σ  weight_i / (k + rank_i + 1)
           i ∈ channels where d appears
```

Where `k=60` is the standard RRF constant that prevents top-ranked items from dominating (a result at rank 1 scores `weight / 61`, not infinitely better than rank 2's `weight / 62`).

Items are deduplicated by a key combining their source type and the first 120 characters of content. Items appearing in multiple channels naturally accumulate score contributions from each channel — a result found by both graph traversal and vector similarity is almost certainly relevant.

**Channel weights applied by HybridGraphRetriever:**

| Channel | Weight | Rationale |
|---------|--------|-----------|
| Graph Local Search | 1.2× | Structural relationships are high-signal for factual queries |
| Graph→Vector Bridge | 1.1× | Entity-linked context is precise by construction |
| Vector Semantic | 1.0× | Baseline semantic relevance |
| Community Global | 0.8× | Broad context; useful but lower specificity |

**Post-fusion score adjustments** are applied at retrieval result level based on document lifecycle status:
- **Confirmed** sources receive a `1.5×` boost to their fused score
- **Deprecated** sources receive a `0.5×` penalty

---

## 14. RAG Methodology Selection & Auto-Fallback

`ComponentFactory` applies several automatic fallbacks before returning components to `RAGLibrary`. These protect against common configuration mistakes and missing optional dependencies without crashing the pipeline.

### Fallback Rules

```python
# In build_components():

# 1. Anthropic has no embedding API
if cfg.embedder == "anthropic":
    warnings.warn("Anthropic has no embedding API. Switching embedder to 'sentence_transformers'.")
    embedder_key = "sentence_transformers"

# 2. Graph store fails to initialize → vector-only
graph_store = _build_graph_store(cfg)  # returns None on any exception
if graph_store is None:
    warnings.warn("Graph store unavailable. Graph features disabled.")

# 3. hybrid_graph without a graph store → naive
if cfg.rag_method == "hybrid_graph" and graph_store is None:
    warnings.warn("hybrid_graph requires a graph store. Falling back to 'naive'.")
    rag_method = "naive"

# 4. nlp_only extraction without spaCy → llm_only
if cfg.extraction == "nlp_only" and nlp_model is None:
    warnings.warn("spaCy unavailable. Falling back to llm_only extraction.")
    → LLMExtractor used instead of NLPExtractor

# 5. OCR provider fails → tesseract
if OCRFactory.create(cfg.ocr, cfg) raises:
    warnings.warn("OCR provider failed. Falling back to tesseract.")
    → TesseractOCR used
```

### Selection Logic Table

| Configured Method / Component | Requires | Auto-fallback |
|-------------------------------|----------|---------------|
| `hybrid_graph` retriever | graph store initialized | → `naive` + UserWarning |
| `microsoft_graphrag` retriever | `graphrag` library installed | → `hybrid_graph` |
| `anthropic` embedder | (no embedding API exists) | → `sentence_transformers` + UserWarning |
| Community search (Channel 3) | vector store with metadata support | → disabled for `faiss` |
| NLP extraction | spaCy installed | → `llm_only` + UserWarning |
| APOC subgraph traversal | APOC Neo4j plugin | → manual Cypher BFS fallback |
| Leiden community detection | Neo4j GDS plugin | → Louvain → skip (3-channel retrieval) |
| Any graph store | provider library installed | → `None` (vector-only) + UserWarning |
| Any optional dependency | package installed | → `ImportError` with install hint |

### Zero-Infrastructure Configuration

The following combination requires zero external infrastructure and runs fully offline:

```python
rag = RAGLibrary(
    embedder="sentence_transformers",   # local model, no API key
    vector_store="faiss",               # local file, no server
    graph_store="networkx",             # in-memory graph, no database
    llm="ollama",                       # local Ollama server
    extraction="nlp_only",              # spaCy only, no LLM API calls
)
```

Cost: zero API calls, zero external servers required. Suitable for air-gapped environments or development without API access.

---

## 15. Incremental Ingestion

The system never re-processes unchanged content.

**Hash-based change detection:**
1. On each `ingest()` call, compute `SHA-256(text.encode("utf-8"))` via `content_hash()`
2. Compare against the stored hash in `HashStore` (backed by `doc_hashes.json`)
3. **Match** → return `{"status": "skipped", "reason": "unchanged"}` immediately, zero API calls
4. **Mismatch** → content changed:
   - Remove old subgraph: `graph_store.remove_doc_subgraph(doc_id)` (deletes relations, orphan entities, `ChunkRef` nodes)
   - Remove old vectors: `vector_store.delete_by_doc_id(doc_id)`
   - Remove stale page index: `page_index.remove(doc_id)`
   - Run the full 9-step extraction pipeline on the new content
   - Update `HashStore` with the new hash

**Garbage collection via `sync()`:**
`IngestionPipeline.sync(current_doc_ids: set[str])` compares the set of doc IDs in `HashStore` against the provided set. Any doc_id in `HashStore` but not in `current_doc_ids` is treated as deleted — `remove_document()` is called for each stale entry. This handles the case where source documents are removed from the source system.

**Why not diff-based updates?** Diffing at the triple level (add new triples, remove deleted ones) is fragile because extraction is non-deterministic — the same text can produce slightly different triples on different runs due to LLM temperature and spaCy model updates. Clean removal and full re-extraction is simpler and guarantees graph consistency.

---

## 16. Knowledge Lifecycle & Confidence Model

### Confidence Scores

Every `Entity` and `Relation` carries a `confidence` float (0.0–1.0) set at extraction time and modified by lifecycle events:

| Source | Default Confidence |
|--------|--------------------|
| spaCy NER entities | 0.85 |
| Dependency-parsed relations | 0.75 |
| SVO triple relations | 0.70 |
| Noun phrase entities | 0.60 |
| LLM-extracted entities | 0.90 |
| LLM-extracted relations | 0.85 |

### Source Lifecycle

Each ingested document is tracked as a `DocumentMeta` node with a `SourceStatus`:

```
  PENDING ──────► CONFIRMED
     │                │
     └───────► DEPRECATED ◄────┘
```

- **Pending** — default for new sources. Confidence values set at extraction. No score modification on retrieval.
- **Confirmed** — `rag.confirm(doc_id)` or `KnowledgeUpdater.confirm_source(doc_id)`. Sets all entity and relation confidences for this source to `1.0`. Retrieval results from confirmed sources receive a `1.5×` score boost after RRF fusion.
- **Deprecated** — `rag.deprecate(doc_id)`. Halves all confidence values for triples from this source. Retrieval results from deprecated sources receive a `0.5×` score penalty. Useful for marking superseded documents (e.g., an outdated report replaced by a newer version) without deleting the knowledge.

### Conflict Detection

`KnowledgeUpdater.detect_conflicts(entity_name)` queries the graph for cases where the same entity has the same relation type pointing to different targets from different sources:

```
Source A: "Anthropic" --[RAISED]--> "$7 Billion"
Source B: "Anthropic" --[RAISED]--> "$10 Billion"
```

This surfaces contradictions for human review rather than silently retaining both. The method returns a list of `{"entity", "relation", "versions": [...]}` dicts.

### Pruning

`KnowledgeUpdater.prune_low_confidence(threshold=0.5)` removes all relations from the graph whose confidence falls below the threshold. Returns the count of relations removed. This is especially useful after deprecating sources — their halved confidence values may drop below 0.5, effectively removing their knowledge from the graph without the overhead of `remove_doc_subgraph()`.

---

## 17. Cost Optimization Strategy

### Embedding Cost

The dominant API cost driver for most workloads. The pipeline batches aggressively:
- `embed_batch(texts)` sends up to 100 texts per API call (provider-configurable)
- A 1,000-chunk corpus costs ~10 Gemini embedding API calls, not 1,000
- Community summaries are embedded in the same batch flow during `build_communities()`

### Extraction Cost

| Strategy | API Calls per Chunk |
|----------|---------------------|
| `llm_only` | 1 LLM call per chunk |
| `nlp_only` | 0 API calls (all local spaCy) |
| `hybrid` (augment mode) | 0 calls for NLP (local); 1 LLM call per chunk with NLP findings |

In `augment` mode, the LLM augmentation prompt instructs the model to find only genuinely new information that NLP missed. Because NLP already extracts the bulk of entities and relations, the LLM's marginal contribution per call is small — the main benefit is capturing implicit associations and causal relationships that spaCy's syntactic analysis cannot detect.

### Incremental Cost

Hash-based skipping means re-running `rag.ingest_dir()` on an unchanged corpus costs **zero API calls**. Only modified or newly added documents incur extraction and embedding costs.

### Zero-Cost Alternatives

- **Lazy imports** — unused providers cost zero import time and zero memory until first use. Installing `raglib` without, e.g., `pinecone-client` is not an error; the `ImportError` only surfaces when `vector_store="pinecone"` is selected.
- **`graph_store="none"`** — eliminates the Neo4j server requirement entirely. The pipeline runs in pure vector mode.
- **`graph_store="networkx"`** — in-memory graph, zero infrastructure cost, zero external server. Suitable for small corpora (tens of thousands of entities) where persistence across sessions is not required.
- **`vector_store="faiss"`** — fastest local vector search, no server required, persists to a local file. Community search is automatically disabled (FAISS has no metadata storage for per-community retrieval).
- **`sentence_transformers` + `networkx` + `faiss`** — fully offline, zero API cost configuration. All computation runs locally.

### Rate Limiting

Built-in rate limiting is applied at the generator and embedder level via configurable RPM limits. The default protects against free-tier quota exhaustion while keeping throughput maximized within quota constraints.

---

## 18. Failure Modes & Graceful Degradation

The pipeline is designed so that losing any single component degrades quality but does not crash the overall pipeline. Each step in `IngestionPipeline.ingest()` wraps its critical operations in `try/except` blocks and logs warnings rather than propagating exceptions.

| Failure | Degradation |
|---------|-------------|
| Neo4j GDS not installed | Leiden unavailable; Louvain attempted; if Louvain also fails, community detection skipped. Retrieval continues on 3 channels (no community channel). |
| APOC plugin not installed | Subgraph traversal falls back to manual Cypher BFS. Less efficient but functionally equivalent. |
| Graph store fails to initialize | `graph_store = None`; `hybrid_graph` → `naive` auto-fallback; all graph channels disabled; vector-only operation. |
| Gemini API rate limit hit | Tenacity retries with exponential backoff (configurable attempts). |
| Any LLM generator API down | `llm_only` and `hybrid` extractors fail per chunk; that chunk's extraction produces an empty `ExtractionResult`. NLP extraction continues independently if `hybrid` mode is active. |
| spaCy not installed | NLP model returns `None`; `NLPExtractor` not built; `hybrid` extractor falls back to `LLMExtractor`; `sentence` chunker falls back to `fixed`. |
| spaCy model not found | `en_core_web_trf` not found → `en_core_web_sm` attempted → auto-download attempted. |
| OCR provider fails to initialize | `OCRFactory` falls back to Tesseract. If Tesseract also unavailable, `ocr = None`; image files produce empty documents. |
| Embedding fails for a chunk | `chunk.embedding` remains `None`; chunk skipped in `vector_store.upsert_chunks()`. Chunk still written to graph. |
| Vector store upsert fails | Warning logged; hash store still updated. Affected chunks are retrievable via graph channel only. |
| `faiss` selected (no metadata) | Community search disabled automatically; 3-channel retrieval (graph + vector + bridge). |
| `networkx` graph store | No persistence between sessions; graph is rebuilt from scratch each time `RAGLibrary` is instantiated. Not suitable for production workloads with large corpora. |
| Optional dependency missing | `ImportError` raised at instantiation time with a message indicating which `pip install` command resolves it. Never raised at library import time. |
| Document hash store corrupted | `HashStore._load()` catches the JSON parse error and returns an empty dict. All documents are treated as new on next ingest. |
| ChromaDB collection corrupted | Re-ingest with `hash_store` cleared (delete `doc_hashes.json`) to force full re-processing. |
