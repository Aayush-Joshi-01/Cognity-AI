<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Neo4j-Online-4581C3?logo=neo4j&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Local-FF6F00" />
  <img src="https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

# HybridGraphRAG

**A production-grade hybrid Retrieval-Augmented Generation system combining knowledge graph traversal, vector semantic search, and Microsoft GraphRAG community intelligence — with NLP-first extraction for cost efficiency.**

---

## Why This Exists

Standard RAG (embed chunks → cosine similarity → generate) loses structural relationships between entities. Pure Graph RAG captures structure but misses semantic nuance. Community-level summarization (Microsoft GraphRAG) handles global queries but is expensive to build.

**HybridGraphRAG fuses all three** — and does it economically by running local NLP (spaCy) for ~70% of knowledge extraction before touching any API.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Documents   │────►│  NLP Layer   │────►│ Gemini Augment│
│  (+ pages)   │     │  (spaCy)     │     │ (only gaps)   │
└─────────────┘     │  • NER       │     └───────┬───────┘
                    │  • SVO       │             │
                    │  • DepParse  │             │
                    │  • Coref     │             │
                    │  • Chunking  │             │
                    └──────┬───────┘             │
                           │          ┌─────────┘
                           ▼          ▼
┌──────────────────────────────────────────────────────┐
│              Ingestion Pipeline                       │
│  Hash check → Page index → Chunk → Extract → Embed   │
└──────────────┬────────────────────────┬──────────────┘
               │                        │
               ▼                        ▼
┌──────────────────┐     ┌──────────────────────┐
│   Neo4j (Online)  │◄──►│   ChromaDB (Local)    │
│  • Entities       │     │  • Chunk embeddings   │
│  • Relations      │     │  • Community embeds   │
│  • Communities    │     │  • Page metadata      │
│  • ChunkRef links │     │  • Entity-linked IDs  │
└──────────────────┘     └──────────────────────┘
               │                        │
               └────────┬───────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│           4-Channel Hybrid Retriever                  │
│  Ch1: Graph Local (BFS subgraph traversal)           │
│  Ch2: Vector Semantic (cosine similarity)            │
│  Ch3: Community Global (GraphRAG summaries)          │
│  Ch4: Graph→Vector Bridge (entity-linked chunks)     │
│                                                       │
│  → Reciprocal Rank Fusion → Source-confidence boost   │
│  → Gemini generation with multi-context prompt        │
└──────────────────────────────────────────────────────┘
```

---

## Key Features

### Intelligent Extraction (NLP-First)
- **Named Entity Recognition** — spaCy transformer NER (Person, Org, Location, Concept, etc.)
- **Subject-Verb-Object triples** — dependency-parsed relation extraction from every sentence
- **Coreference resolution** — pronoun→entity linking ("He founded" → "Sam Altman founded")
- **Noun phrase entities** — catches compound concepts NER misses ("Constitutional AI", "protein folding")
- **Gemini augmentation** — LLM fills only the semantic gaps NLP can't catch (causal links, implicit associations)

### Microsoft GraphRAG Patterns
- **Leiden community detection** via Neo4j GDS
- **Hierarchical community summarization** — Gemini generates title + summary per community
- **Global search** — community summaries answer broad thematic queries
- **Community embeddings** — stored in dedicated ChromaDB collection for vector-based community retrieval

### 4-Channel Hybrid Retrieval
| Channel | What It Does | Weight |
|---------|-------------|--------|
| **Graph Local** | BFS subgraph expansion from seed entities | 1.2× |
| **Graph→Vector Bridge** | Entity `MENTIONED_IN` edges fetch linked chunks | 1.1× |
| **Vector Semantic** | Cosine similarity on chunk embeddings | 1.0× |
| **Community Global** | Ranked community summaries for thematic context | 0.8× |

Results are fused via **Reciprocal Rank Fusion (RRF)** with confirmed-source boosting.

### Incremental & Economical
- **SHA-256 hash check** — unchanged docs are skipped entirely
- **Stale data cleanup** — on re-ingest, old subgraph + vectors are removed before new data
- **Page/section indexing** — structural awareness with persistent page index
- **Batch embeddings** — 100 texts per API call
- **Rate limiting** — built into every Gemini call
- **Configurable extraction mode** — `"augment"` (NLP + LLM) or `"full"` (LLM only)

### Knowledge Lifecycle
- **Confirm sources** → boosts all triples from that doc to confidence 1.0
- **Deprecate sources** → halves confidence of associated triples
- **Conflict detection** → finds same `(entity, relation)` pointing to different targets across sources
- **Low-confidence pruning** → removes triples below threshold
- **Health reporting** → entity/relation/doc/community counts + average confidence

---

## Quick Start

### Prerequisites

| Component | Purpose | Setup |
|-----------|---------|-------|
| **Neo4j Aura** (or self-hosted) | Knowledge graph | [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) — free tier works |
| **Neo4j APOC plugin** | Subgraph traversal | Usually pre-installed on Aura |
| **Neo4j GDS plugin** | Community detection (Leiden) | Required for GraphRAG communities; gracefully skips if absent |
| **Google Gemini API key** | Embeddings + augmentation + generation | [aistudio.google.com](https://aistudio.google.com/) |
| **Python 3.10+** | Runtime | — |

### Installation

```bash
git clone https://github.com/yourname/hybrid-graph-rag.git
cd hybrid-graph-rag

pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_trf    # Best accuracy (~500MB)
python -m spacy download en_core_web_sm     # Lightweight fallback (~12MB)
```

### Environment Variables

```bash
export NEO4J_URI="neo4j+s://xxxxxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export GEMINI_API_KEY="your-gemini-api-key"

# Optional
export NEO4J_DATABASE="neo4j"
export CHROMA_PERSIST_DIR="./chroma_store"
```

### Basic Usage

```python
from main import build_pipeline

# Wire all components
c = build_pipeline()
pipeline = c["pipeline"]
retriever = c["retriever"]
updater = c["updater"]

# ── Ingest documents ────────────────────────────────
pipeline.ingest(
    doc_id="doc_001",
    text="OpenAI was founded by Sam Altman in 2015...",
    source_name="tech_report",
    status="confirmed",      # or "pending"
)

# Batch ingest
pipeline.ingest_batch([
    {"doc_id": "doc_002", "text": "...", "source_name": "research"},
    {"doc_id": "doc_003", "text": "...", "source_name": "notes"},
])

# ── Build communities (run after ingestion) ─────────
pipeline.build_communities()

# ── Query ───────────────────────────────────────────
answer = retriever.query("Who founded OpenAI?")

# Query with full source attribution
result = retriever.query_with_sources("Compare AI safety approaches")
print(result["answer"])
print(result["sources"])          # graph, vector, community sources
print(result["retrieval_scores"]) # per-result scores by channel
print(result["seed_entities"])    # NLP-extracted query entities

# ── Knowledge lifecycle ─────────────────────────────
updater.confirm_source("doc_002")
updater.deprecate_source("doc_old")
conflicts = updater.detect_conflicts("San Francisco")
updater.prune_low_confidence(threshold=0.3)
print(updater.health_report())
```

### Run the Demo

```bash
python main.py
```

---

## Configuration

All settings live in `config.py` as dataclasses. Override via constructor or environment variables.

```python
from config import Config, NLPConfig, IngestionConfig

config = Config(
    nlp=NLPConfig(
        spacy_model="en_core_web_sm",         # Use lighter model
        semantic_chunk_sentences=8,            # Bigger chunks
    ),
    ingestion=IngestionConfig(
        gemini_extraction_mode="full",         # Skip NLP, use Gemini only
        max_gemini_chunks_per_doc=100,
        confidence_threshold=0.6,
    ),
)
```

| Setting | Default | Description |
|---------|---------|-------------|
| `nlp.spacy_model` | `en_core_web_trf` | spaCy model (trf=best, sm=fast) |
| `nlp.semantic_chunk_sentences` | `5` | Sentences per chunk |
| `nlp.semantic_chunk_overlap` | `1` | Sentence overlap between chunks |
| `ingestion.gemini_extraction_mode` | `"augment"` | `"augment"` = NLP+LLM, `"full"` = LLM only |
| `ingestion.confidence_threshold` | `0.5` | Pruning cutoff |
| `ingestion.confirmed_boost` | `1.5` | Score multiplier for confirmed sources |
| `graphrag.leiden_resolution` | `1.0` | Community detection granularity |
| `graphrag.max_community_levels` | `3` | Hierarchical depth |
| `graphrag.global_search_top_communities` | `5` | Communities returned in global search |
| `gemini.batch_embed_limit` | `100` | Texts per embedding API call |
| `gemini.rpm_limit` | `15` | Rate limit (requests/minute) |

---

## Project Structure

```
hybrid_rag/
├── main.py                 # Entry point, wiring, demo
├── config.py               # All configuration dataclasses
├── models.py               # Pydantic data models
├── nlp_processor.py        # spaCy NLP pipeline (NER, SVO, coref, chunking)
├── gemini_extractor.py     # Gemini API layer (augment, describe, summarize, embed)
├── graph_manager.py        # Neo4j CRUD, communities, retrieval, lifecycle
├── vector_manager.py       # ChromaDB dual-collection (chunks + communities)
├── page_index.py           # Persistent page/section index
├── ingestion.py            # Full ingestion pipeline orchestration
├── hybrid_retriever.py     # 4-channel retrieval + RRF + generation
├── knowledge_updater.py    # Source lifecycle + conflict detection
└── requirements.txt
```

---

## Cost Analysis

For a **1000-document corpus** (avg 2000 tokens each):

| Operation | Without NLP-First | With NLP-First | Savings |
|-----------|-------------------|----------------|---------|
| Entity extraction | ~4000 Gemini calls | ~4000 spaCy (free) + ~1000 Gemini augment | **~75%** |
| Embeddings | ~4000 calls | ~40 batched calls (100/batch) | **~99%** |
| Community summaries | ~50 calls | ~50 calls | Same |
| Re-ingestion (no changes) | ~4000 calls | **0 calls** (hash skip) | **100%** |

---

## Neo4j Graph Schema

```
(:Entity {name, entity_type, description, confidence, mentions, source_id, extraction_method})
(:DocumentMeta {doc_id, source_name, content_hash, status, chunk_count, entity_count})
(:Community {community_id, title, summary, rank, level, entity_count})
(:ChunkRef {chunk_id, doc_id})

(:Entity)-[:MENTIONED_IN]->(:ChunkRef)
(:Entity)-[:BELONGS_TO_COMMUNITY]->(:Community)
(:Entity)-[:<DYNAMIC_RELATION> {confidence, weight, description, source_id}]->(:Entity)
```

---

## License

MIT

---

## Contributing

PRs welcome. Key areas for contribution:
- Additional NLP backends (Stanza, Flair)
- Async ingestion pipeline
- Streaming retrieval
- Alternative LLM backends (OpenAI, local models)
- Web UI for knowledge graph exploration
