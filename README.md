# cognity-ai

**Modular, provider-agnostic RAG library — any LLM, any vector store, any graph DB, any file format.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License: Apache 2.0"/>
  <img src="https://img.shields.io/badge/Version-2.1.0-blue" alt="Version: 2.1.0"/>
  <img src="https://img.shields.io/badge/Neo4j-supported-blue?logo=neo4j&logoColor=white" alt="Neo4j"/>
  <img src="https://img.shields.io/badge/ChromaDB-supported-orange" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Gemini-supported-4285F4?logo=google&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/spaCy-supported-09A3D5" alt="spaCy"/>
  <img src="https://img.shields.io/badge/OpenAI-supported-000000?logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/Anthropic-supported-7C3AED" alt="Anthropic"/>
</p>

---

## Overview

`cognity-ai` is a drop-in RAG (Retrieval-Augmented Generation) service for AI agents. It was extracted and redesigned from the original `hybrid_rag` monolith into a fully modular library — every component is swappable at runtime with zero code changes beyond configuration.

**What makes it different:**

- **Any LLM:** Gemini, OpenAI, Azure OpenAI, Anthropic, AWS Bedrock, Cohere, Ollama, Vertex AI
- **Any embedder:** Same provider list — Anthropic automatically falls back to `sentence-transformers` since it has no native embedding API
- **Any vector store:** ChromaDB, Qdrant, Pinecone, FAISS, Weaviate, Milvus, pgvector, Azure AI Search
- **Any graph DB:** Neo4j, Memgraph, ArangoDB, NetworkX (in-memory), Microsoft GraphRAG
- **Any file format:** PDF, DOCX, XLSX, PPTX, CSV, HTML, JSON, YAML, TXT, MD, images (via multimodal OCR)
- **Multiple RAG methodologies:** `hybrid_graph`, `naive`, `vector_only`, `graph_only`, `parent_child`, `multi_query`, `microsoft_graphrag`, `adaptive`
- **Smart defaults:** The best available methodology is automatically selected based on which stores are configured

The primary API surface is a single class — `RAGLibrary` — which wires up the full pipeline from ingestion through retrieval and generation.

---

## Architecture Overview

```
Files (PDF / DOCX / XLSX / PPTX / images / ...)
        |
        v
   [ Loaders ]  ──► OCR (if image: Gemini Vision / GPT-4o / Claude / Tesseract)
        |
        v
   [ PageIndex ]  (regex / structural / hybrid page boundary detection)
        |
        v
   [ Chunkers ]  (sentence / fixed / recursive / semantic / parent_child / hybrid)
        |
        v
   [ Extractors ]  (NLP + LLM hybrid entity & relation extraction)
        |
        v
   [ Embedders ]  (Gemini / OpenAI / Bedrock / Cohere / Ollama / SentenceTransformers)
        |
        v
  ┌─────────────────────┐
  │   Graph Store       │   ◄── Neo4j / Memgraph / ArangoDB / NetworkX / MS GraphRAG
  │   Vector Store      │   ◄── ChromaDB / Qdrant / Pinecone / FAISS / Weaviate / ...
  └─────────────────────┘
        |
        v
   [ Retrievers ]
   4-channel Hybrid:
     ├── Graph BFS traversal
     ├── Vector similarity search
     ├── Community summary search
     └── Bridge node discovery
          └──► RRF fusion
        |
        v
   [ Generators ]  (Gemini / OpenAI / Anthropic / Bedrock / Cohere / Ollama / Vertex AI)
        |
        v
      Answer
```

---

## Quick Start

### Installation

```bash
# Default: Gemini + Neo4j + ChromaDB + spaCy + all loaders
pip install -e ".[default]"

# Selective extras — mix and match
pip install -e ".[openai,qdrant,pdf]"
pip install -e ".[anthropic,pinecone,office]"
pip install -e ".[bedrock,faiss]"

# Everything
pip install -e ".[all]"

# spaCy language model (required for NLP-based extraction)
python -m spacy download en_core_web_trf   # best accuracy (~500 MB)
python -m spacy download en_core_web_sm    # lightweight (~12 MB)
```

### Zero-config start

```python
from cognity_ai import RAGLibrary

rag = RAGLibrary(gemini_api_key="...", neo4j_password="...")

# Ingest any file format — format is auto-detected from extension
rag.ingest("report.pdf")
rag.ingest("data.xlsx")
rag.ingest("slides.pptx")
rag.ingest("photo.jpg")          # OCR via Gemini Vision
rag.ingest_dir("./docs/")        # recursive, all supported formats

# Optional: build GraphRAG community summaries for global search
rag.build_communities()

answer = rag.query("What are the key findings?")

result = rag.query_with_sources("Who founded Anthropic?")
print(result["answer"])
print(result["sources"])
```

### Full explicit configuration

```python
rag = RAGLibrary(
    rag_method="hybrid_graph",
    chunker="sentence",
    embedder="openai",
    vector_store="qdrant",
    graph_store="neo4j",
    llm="anthropic",
    ocr="gemini_vision",
    page_index="hybrid",
    openai_api_key="...",
    anthropic_api_key="...",
    neo4j_uri="bolt://localhost:7687",
    neo4j_password="...",
)
```

Every parameter has a sensible default. You only need to set the keys for the providers you actually use.

---

## Supported File Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | `.pdf` | pdfplumber + pypdf; page-aware; extracts embedded images |
| Word | `.docx` | python-docx; tables, headings, embedded images |
| Excel | `.xlsx`, `.xls` | openpyxl + pandas; ingested per-sheet |
| PowerPoint | `.pptx` | python-pptx; slides + speaker notes + images |
| CSV / TSV | `.csv`, `.tsv` | pandas; auto-detects delimiter |
| HTML | `.html`, `.htm` | beautifulsoup4; strips tags, extracts text |
| Text / Markdown | `.txt`, `.md` | native; markdown heading detection |
| JSON / YAML | `.json`, `.yaml`, `.yml` | recursive key-value flattening |
| Images | `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`, `.webp` | multimodal OCR (see below) |

---

## OCR Providers

Image files are processed by a configurable OCR provider. The default is Gemini Vision, which handles complex layouts, mixed text/diagram pages, and handwriting well.

| Provider | Key | Method |
|----------|-----|--------|
| Gemini Vision (default) | `gemini_vision` | Gemini 2.0 Flash multimodal |
| OpenAI Vision | `openai_vision` | GPT-4o vision |
| Anthropic Vision | `anthropic_vision` | Claude 3.5 Sonnet vision |
| Azure Vision | `azure_vision` | Azure-deployed GPT-4o vision |
| Bedrock Vision | `bedrock_vision` | AWS Bedrock Claude vision |
| Tesseract | `tesseract` | Local pytesseract (fully offline) |

Configure via:

```python
rag = RAGLibrary(ocr="tesseract")                                         # offline
rag = RAGLibrary(ocr="anthropic_vision", anthropic_api_key="...")
```

---

## 🎬 Multimodal RAG (Experimental)

cognity-ai includes an experimental `cognity-ai.multimodal` subpackage for Image, Video, and Audio RAG using multimodal embedders.

### Multimodal Embedders

| Key | Class | Dims | Modalities | Install |
|---|---|---|---|---|
| CLIP | `CLIPEmbedder` | 512/768 | Image + Text | `pip install cognity-ai[clip]` |
| SigLIP | `SigLIPEmbedder` | 768/1024 | Image + Text | `pip install cognity-ai[siglip]` |
| ImageBind | `ImageBindEmbedder` | 1024 | Image + Text + Audio + Video | See [ImageBind](https://github.com/facebookresearch/ImageBind) |
| BLIP-2 | `BLIP2Embedder` | 256 | Image → Caption | `pip install cognity-ai[blip2]` |

### Image RAG

```python
from cognity_ai.multimodal import ImageIngestionPipeline, ImageRetriever
from cognity_ai.multimodal.embedders import CLIPEmbedder
from cognity_ai.multimodal.stores import ChromaMultimodalStore

embedder = CLIPEmbedder()
store = ChromaMultimodalStore()

# Ingest images
pipeline = ImageIngestionPipeline(embedder=embedder, store=store)
pipeline.ingest("photo.jpg")
pipeline.ingest_batch(["img1.png", "img2.jpg"])

# Text-to-image retrieval
retriever = ImageRetriever(embedder=embedder, store=store)
results = retriever.retrieve("a dog playing in a park", top_k=5)
for r in results:
    print(r.chunk_id, r.score, r.metadata.get("caption"))
```

### Video RAG

```python
from cognity_ai.multimodal import VideoIngestionPipeline, VideoRetriever
from cognity_ai.multimodal.embedders import CLIPEmbedder
from cognity_ai.multimodal.transcribers import WhisperLocalTranscriber
from cognity_ai.multimodal.stores import ChromaMultimodalStore

embedder = CLIPEmbedder()
transcriber = WhisperLocalTranscriber(model_size="base")
store = ChromaMultimodalStore()

# Ingest video (extracts frames + transcribes audio)
pipeline = VideoIngestionPipeline(
    embedder=embedder,
    store=store,
    transcriber=transcriber,
)
pipeline.ingest("lecture.mp4")

# Retrieve relevant video segments with timestamps
retriever = VideoRetriever(embedder=embedder, store=store)
results = retriever.retrieve("neural networks training process")
for r in results:
    chunk = r.video_chunk
    print(f"At {chunk.start_ms//1000}s – {chunk.end_ms//1000}s: {chunk.transcript[:100]}")
```

### Audio RAG

```python
from cognity_ai.multimodal import AudioIngestionPipeline, AudioRetriever
from cognity_ai.multimodal.transcribers import WhisperLocalTranscriber
from cognity_ai.multimodal.stores import ChromaMultimodalStore
from cognity_ai.embedders import GeminiEmbedder

transcriber = WhisperLocalTranscriber(model_size="small")
store = ChromaMultimodalStore()
text_embedder = GeminiEmbedder(api_key="...")

pipeline = AudioIngestionPipeline(
    transcriber=transcriber,
    store=store,
    text_embedder=text_embedder,
)
pipeline.ingest("podcast.mp3")

retriever = AudioRetriever(embedder=None, store=store, text_embedder=text_embedder)
results = retriever.retrieve("discussion about machine learning")
```

### Cross-Modal Retrieval (ImageBind)

```python
from cognity_ai.multimodal.retrievers import CrossModalRetriever
from cognity_ai.multimodal.embedders import ImageBindEmbedder

# Query with text → find relevant images, video clips, AND audio segments
retriever = CrossModalRetriever(
    embedder=ImageBindEmbedder(),
    store=store,
    search_modalities=["image", "video", "audio"],
)
results = retriever.retrieve("product launch announcement")
# Returns mixed: ImageChunks, VideoChunks, AudioChunks unified by score
```

### Transcription Providers

| Provider | Class | Install |
|---|---|---|
| Whisper (local) | `WhisperLocalTranscriber` | `pip install cognity-ai[whisper]` |
| OpenAI Whisper API | `WhisperAPITranscriber` | `pip install openai` |
| Google Speech-to-Text | `GoogleSTTTranscriber` | `pip install google-cloud-speech` |
| AWS Transcribe | `AWSTranscribeTranscriber` | `pip install boto3` |

### Install

```bash
pip install cognity-ai[clip]          # Image RAG with CLIP
pip install cognity-ai[siglip]        # Image RAG with SigLIP (higher quality)
pip install cognity-ai[video]         # Video support (frame extraction, scene detection)
pip install cognity-ai[audio]         # Audio loading
pip install cognity-ai[whisper]       # Local Whisper transcription
pip install cognity-ai[multimodal]    # Everything above
```

> **Note:** ImageBind requires manual installation. See the [ImageBind GitHub](https://github.com/facebookresearch/ImageBind) for instructions.

---

## RAG Methodologies

| Method | Description | When to Use |
|--------|-------------|-------------|
| `hybrid_graph` (default) | 4-channel retrieval: Graph BFS + Vector + Community + Bridge nodes, fused with RRF | Knowledge graphs, multi-hop reasoning, structured corpora |
| `naive` | Pure vector cosine similarity, no graph | Quick setup, unstructured flat text |
| `vector_only` | Vector similarity + community summary search | No graph store, but communities exist |
| `graph_only` | Graph traversal only, no vector lookup | Structured knowledge bases with clear entity relationships |
| `parent_child` | Retrieve small precise chunks, return their larger parent context | Long documents where context window matters |
| `multi_query` | Generate N query variants, merge and deduplicate results | Complex or ambiguous queries |
| `microsoft_graphrag` | Official MS GraphRAG local + global search modes | Microsoft ecosystem integrations |
| `adaptive` | Auto-routes to the best method based on query classification | Unknown or mixed query patterns |

### Per-query method override

```python
# Override method for a single query without reconfiguring the library
answer = rag.query("What themes emerge across all documents?", method="multi_query")
result = rag.query_with_sources("Who founded Anthropic?", method="hybrid_graph")
```

---

## Provider Matrix

### LLMs and Embedders

| Provider | Key | Generator | Embedder | Notes |
|----------|-----|-----------|----------|-------|
| Gemini (default) | `gemini` | Yes | Yes | Gemini 2.0 Flash / text-embedding-004 |
| Vertex AI | `vertex_ai` | Yes | Yes | Gemini 1.5 Pro / text-embedding-005 |
| OpenAI | `openai` | Yes | Yes | GPT-4o / text-embedding-3-small |
| Azure OpenAI | `azure_openai` | Yes | Yes | Azure-deployed GPT-4o and embedding models |
| Anthropic | `anthropic` | Yes | No | claude-3-5-sonnet; embedding falls back to `sentence_transformers` |
| AWS Bedrock | `bedrock` | Yes | Yes | Claude / Titan / Llama + Titan Embeddings V2 |
| Cohere | `cohere` | Yes | Yes | Command R+ / embed-english-v3.0 |
| Ollama | `ollama` | Yes | Yes | llama3, mistral, nomic-embed-text (fully local) |
| SentenceTransformers | `sentence_transformers` | No | Yes | all-MiniLM-L6-v2 (offline, no API key needed) |

> **Note on Anthropic embeddings:** Anthropic does not provide an embedding API. When `llm="anthropic"` and no explicit `embedder` is set, `cognity-ai` automatically falls back to `sentence_transformers` for embeddings.

---

## Vector Stores

| Store | Key | Type |
|-------|-----|------|
| ChromaDB (default) | `chroma` | Local persistent |
| Qdrant | `qdrant` | Local or Qdrant Cloud |
| Pinecone | `pinecone` | Cloud (serverless or pod) |
| FAISS | `faiss` | Local in-memory |
| Weaviate | `weaviate` | Local or Weaviate Cloud |
| Milvus | `milvus` | Local or Zilliz Cloud |
| pgvector | `pgvector` | PostgreSQL extension |
| Azure AI Search | `azure_search` | Azure cloud |

---

## Graph Stores

| Store | Key | Type |
|-------|-----|------|
| Neo4j (default) | `neo4j` | Dedicated graph DB (Bolt protocol) |
| Microsoft GraphRAG | `microsoft_graphrag` | Wraps the official `graphrag` library |
| Memgraph | `memgraph` | Open source, Neo4j-compatible (Bolt) |
| ArangoDB | `arangodb` | Multi-model (document + graph) |
| NetworkX | `networkx` | In-memory Python graph (testing / no-DB mode) |

---

## AI Observability & Token Tracking

Every generator call is automatically timed and its token usage extracted from the native API response — no additional configuration required.

```python
from cognity_ai import RAGLibrary, LoggingObserver

# Attach the built-in logging observer
rag = RAGLibrary(
    gemini_api_key="...",
    observer=LoggingObserver(),  # emits JSON to cognity_ai.observability logger
)

answer = rag.query("What are the findings?")

# Inspect aggregated usage
print(rag.token_summary())
# {
#   "total_generation_calls": 1,
#   "total_prompt_tokens": 312,
#   "total_completion_tokens": 87,
#   "total_tokens": 399,
#   "token_count_source": "native",
#   ...
# }

# Inspect the last N events
for event in rag.observability.recent_events(10):
    print(event.model_dump())
```

### Token counting priority chain

1. **Native** — extracted directly from the API response object (Gemini `usage_metadata`, OpenAI `usage`, Anthropic `message.usage`, Cohere `meta.tokens`, Ollama `eval_count`, Bedrock body)
2. **tiktoken** — used as fallback only when `tiktoken` is installed *and* the model is known to it
3. **Estimate** — word-count heuristic; always available, never fails

### Custom observers

Extend `BaseObserver` to integrate with any observability backend:

```python
from cognity_ai import BaseObserver, GenerationEvent

class MyPrometheusObserver(BaseObserver):
    def on_generation(self, event: GenerationEvent) -> None:
        generation_counter.inc()
        token_gauge.set(event.token_usage.total_tokens)

rag = RAGLibrary(..., observer=MyPrometheusObserver())
```

The same pattern works for OpenTelemetry spans, Langfuse traces, Datadog metrics, etc.

---

## Entity Autocomplete (Trie)

```python
# After ingesting documents, suggest entity names by prefix
suggestions = rag.suggest_entities("anth")
# → ["Anthropic", "Anthony Hopkins", ...]

# Or use the Trie directly
from cognity_ai import Trie, EntityTrie

t = Trie()
for w in ["apple", "application", "apply", "apt"]:
    t.insert(w)

t.words_with_prefix("app")       # ["apple", "application", "apply"]
t.longest_prefix_match("applet") # "apple"
t.autocomplete("ap", max_results=3)

et = EntityTrie()
et.insert_entity("OpenAI")
et.search_entities("open")  # ["OpenAI"]  — original casing preserved
```

---

## Knowledge Lifecycle Management

`cognity-ai` tracks confidence scores for every extracted knowledge triple. Use the lifecycle API to manage knowledge quality over time.

```python
# Boost confidence for a confirmed, authoritative source
rag.confirm("doc_001")

# Penalize an outdated or superseded document — halves confidence,
# reduces retrieval score for all associated triples
rag.deprecate("old_doc")

# Find contradictions: returns triples that conflict with other sources
conflicts = rag.detect_conflicts("Anthropic")

# Remove low-confidence triples from both stores
rag.prune(threshold=0.5)

# Summarise store health: triple count, avg confidence, conflict rate
print(rag.health_report())
```

---

## Plugin System

Every component type is pluggable. Register custom implementations at runtime and they become available via their key string, just like built-in providers.

```python
from cognity_ai.loaders.base import BaseLoader
from cognity_ai.models.document import Document

class MyLoader(BaseLoader):
    def load(self, path: str) -> list[Document]:
        ...  # parse your custom format here

    @property
    def supported_extensions(self) -> list[str]:
        return [".myext"]


rag.register_loader(".myext", MyLoader)
rag.register_embedder("my_embedder", MyEmbedder)
rag.register_retriever("my_method", MyRetriever)

# Inspect all registered components
print(rag.available_plugins())
```

The same pattern applies to generators, chunkers, extractors, OCR providers, and stores.

---

## Project Structure

```
D:\Graph-RAG\
├── cognity-ai/                        # Main library package
│   ├── library.py                 # RAGLibrary — the primary public API
│   ├── factory.py                 # Component wiring + provider auto-fallback logic
│   ├── registry.py                # Plugin registry for all component types
│   ├── models/                    # Core data models
│   │   ├── document.py            # Document, Chunk, PageInfo
│   │   ├── knowledge.py           # Entity, Relation, Triple, Community
│   │   └── retrieval.py           # RetrievalResult, SourceReference
│   ├── config/                    # Configuration dataclasses
│   │   ├── base.py                # LibraryConfig
│   │   └── providers.py           # Per-provider config (Neo4jConfig, etc.)
│   ├── loaders/                   # File format loaders
│   │   ├── pdf.py                 # pdfplumber + pypdf
│   │   ├── docx.py                # python-docx
│   │   ├── excel.py               # openpyxl + pandas
│   │   ├── pptx.py                # python-pptx
│   │   ├── csv.py                 # pandas
│   │   ├── html.py                # beautifulsoup4
│   │   ├── text.py                # plain text + markdown
│   │   ├── json_loader.py         # JSON + YAML
│   │   ├── image.py               # delegates to OCR provider
│   │   └── factory.py             # extension → loader routing
│   ├── ocr/                       # OCR providers
│   │   ├── gemini_vision.py
│   │   ├── openai_vision.py
│   │   ├── anthropic_vision.py
│   │   ├── azure_vision.py
│   │   ├── bedrock_vision.py
│   │   └── tesseract.py
│   ├── chunkers/                  # Text splitting strategies
│   │   ├── sentence.py
│   │   ├── fixed.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   ├── parent_child.py
│   │   └── hybrid.py
│   ├── page_index/                # Page boundary detection
│   │   ├── regex_index.py
│   │   ├── structural_index.py
│   │   └── hybrid_index.py
│   ├── extractors/                # Entity + relation extraction
│   │   ├── nlp.py                 # spaCy NER + dependency parsing
│   │   ├── llm.py                 # LLM-guided extraction
│   │   └── hybrid.py              # NLP first, LLM gap-fill
│   ├── embedders/                 # Embedding providers
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   ├── azure_openai.py
│   │   ├── vertex_ai.py
│   │   ├── bedrock.py
│   │   ├── cohere.py
│   │   ├── ollama.py
│   │   └── sentence_transformers.py
│   ├── generators/                # LLM response generators
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   ├── azure_openai.py
│   │   ├── anthropic.py
│   │   ├── vertex_ai.py
│   │   ├── bedrock.py
│   │   ├── cohere.py
│   │   └── ollama.py
│   ├── stores/
│   │   ├── vector/                # Vector store adapters
│   │   │   ├── chroma.py
│   │   │   ├── qdrant.py
│   │   │   ├── pinecone.py
│   │   │   ├── faiss.py
│   │   │   └── ...
│   │   └── graph/                 # Graph store adapters
│   │       ├── neo4j.py
│   │       ├── memgraph.py
│   │       ├── arangodb.py
│   │       └── networkx.py
│   ├── retrievers/                # Retrieval strategies
│   │   ├── hybrid_graph.py        # 4-channel + RRF fusion
│   │   ├── naive.py
│   │   ├── vector_only.py
│   │   ├── graph_only.py
│   │   ├── parent_child.py
│   │   ├── multi_query.py
│   │   ├── microsoft_graphrag.py
│   │   └── adaptive.py
│   ├── pipeline/                      # Orchestration
│   │   ├── ingestion.py
│   │   └── knowledge_updater.py
│   ├── observability/                 # NEW: AI observability & token tracking
│   │   ├── models.py                  # TokenUsage, GenerationEvent, etc.
│   │   ├── token_tracker.py           # tiktoken / Estimate chain
│   │   ├── base_observer.py           # ABC for OTEL, Langfuse, etc.
│   │   ├── noop_observer.py           # Zero-overhead default
│   │   ├── logging_observer.py        # JSON log emitter
│   │   └── collector.py               # Fan-out + ring buffer
│   └── utils/                         # NEW: Shared utilities
│       ├── trie.py                    # O(k) prefix lookup
│       ├── hash.py                    # SHA-256 change detection
│       └── token_counter.py           # estimate_tokens() heuristic
├── tests/                             # NEW: Full pytest suite (316 pass, 49 skip)
├── hybrid_rag/                        # DEPRECATED legacy package
├── pyproject.toml                     # Packaging + optional dependency groups
├── requirements.txt                   # Default install dependencies
└── docs/                              # GitHub Pages documentation site
```

---

## Configuration Reference

The `LibraryConfig` dataclass (and the `RAGLibrary` constructor kwargs) accept the following top-level keys:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rag_method` | `str` | `"hybrid_graph"` | Retrieval methodology |
| `chunker` | `str` | `"sentence"` | Text chunking strategy |
| `embedder` | `str` | `"gemini"` | Embedding provider key |
| `vector_store` | `str` | `"chroma"` | Vector store key |
| `graph_store` | `str` | `"neo4j"` | Graph store key |
| `llm` | `str` | `"gemini"` | Generator LLM key |
| `extraction` | `str` | `"hybrid"` | Knowledge extraction strategy (`nlp`, `llm`, `hybrid`) |
| `ocr` | `str` | `"gemini_vision"` | OCR provider for images |
| `page_index` | `str` | `"hybrid"` | Page boundary detection strategy |
| `observer` | `BaseObserver` | `None` | Observability observer instance |
| `observability_config` | `ObservabilityConfig` | `None` | Observability settings |
| `config` | `LibraryConfig` | `None` | Full config override (bypasses all other kwargs) |

Provider-specific settings (API keys, URIs, model names) are passed as additional kwargs and are forwarded to the relevant provider config automatically.

### `MinimalLibraryConfig` — local-only preset

Use `MinimalLibraryConfig` in tests, CI, or offline environments. It uses only components that need zero external services (FAISS, NetworkX, fixed chunker):

```python
from cognity_ai.config.base import MinimalLibraryConfig
from cognity_ai import RAGLibrary

cfg = MinimalLibraryConfig(vector_store="faiss", graph_store="networkx")
rag = RAGLibrary(config=cfg)  # no API keys required for structural use
```

---

## Contributing

Pull requests are welcome. High-priority areas:

- **New loaders:** EPUB, XML, Markdown front-matter, audio transcripts
- **New graph / vector store adapters:** Weaviate, Milvus, pgvector, ArangoDB
- **Streaming retrieval:** async generator interface for token-by-token output
- **Async pipeline:** full async/await ingestion and retrieval path
- **Web UI:** graph exploration and document management interface
- **Evaluation harness:** RAGAS / ARES integration for automated quality scoring

Please open an issue before starting large changes to align on design direction.

---

## License

**Apache 2.0** — see [LICENSE](LICENSE) for full terms.

You may freely use this library as a dependency (`pip install cognity-ai`) in any project, commercial or otherwise. If you incorporate the source code directly into your own project, Apache 2.0 requires that you retain the copyright notice, state what you changed, and include a copy of the license.

> **Why Apache 2.0 and not MIT?**
> Apache 2.0 adds an explicit patent grant (MIT does not), requires attribution notices when source code is copied, and mandates that modified files carry a change notice. For a library of this scope it provides stronger author protection while remaining fully enterprise-friendly — it is the license used by Kubernetes, TensorFlow, FastAPI, and most modern ML tooling.
