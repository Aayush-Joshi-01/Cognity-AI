# Architecture & Concepts

Deep dive into the design decisions, algorithms, and data flow patterns powering HybridGraphRAG.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [The Extraction Pipeline](#2-the-extraction-pipeline)
3. [Semantic Chunking Strategy](#3-semantic-chunking-strategy)
4. [Page Index System](#4-page-index-system)
5. [Graph Data Model](#5-graph-data-model)
6. [Microsoft GraphRAG: Community Detection & Summarization](#6-microsoft-graphrag-community-detection--summarization)
7. [The Graph↔Vector Bridge](#7-the-graphvector-bridge)
8. [4-Channel Hybrid Retrieval](#8-4-channel-hybrid-retrieval)
9. [Reciprocal Rank Fusion (RRF)](#9-reciprocal-rank-fusion-rrf)
10. [Incremental Ingestion](#10-incremental-ingestion)
11. [Knowledge Lifecycle & Confidence Model](#11-knowledge-lifecycle--confidence-model)
12. [Cost Optimization Strategy](#12-cost-optimization-strategy)
13. [Failure Modes & Graceful Degradation](#13-failure-modes--graceful-degradation)

---

## 1. Design Philosophy

The system is built on three core principles:

**Extract locally, augment remotely.** spaCy handles the deterministic, well-understood NLP tasks (NER, dependency parsing, SVO extraction) at zero API cost. Gemini only handles what requires semantic reasoning — implicit associations, causal links, and descriptions that need world knowledge. This isn't just a cost optimization; local NLP is also faster and more predictable.

**Structure enables reasoning.** A knowledge graph captures _relationships_ that flat vector embeddings lose. "Sam Altman founded OpenAI" and "OpenAI developed GPT-4" are two separate chunks in vector space, but a single 2-hop traversal in a graph. The graph makes multi-hop reasoning possible without relying on the LLM to infer connections from concatenated text.

**Multiple retrieval channels beat any single one.** No single retrieval method dominates across all query types. Factual lookups favor graph traversal. Semantic similarity queries favor vector search. Broad thematic questions favor community summaries. The system runs all channels in parallel and lets RRF sort out which signals matter for each query.

---

## 2. The Extraction Pipeline

Extraction happens in two phases: **local NLP** then **LLM augmentation**.

### Phase 1: spaCy NLP (Free, Local)

Four extraction techniques run on every chunk:

**Named Entity Recognition (NER)**
spaCy's transformer-based NER identifies entities and maps them to an ontology:

```
PERSON → Person       ORG → Organization     GPE/LOC → Location
PRODUCT → Product     EVENT → Event          NORP → Group
WORK_OF_ART → CreativeWork    DATE/TIME → Temporal
```

Entities are normalized to title case. Pure numeric/temporal entities shorter than 4 characters are filtered out (they're noise for knowledge graphs).

**Dependency-Based Relation Extraction**
For each sentence containing 2+ named entities, the system walks the dependency parse tree between entity root tokens to infer a relation. The walk follows parent-child edges up to 4 hops, mapping dependency arcs to semantic relations:

```
nsubj → SUBJECT_OF     dobj → ACTS_ON      pobj → RELATES_TO
attr → IS_A            appos → ALSO_KNOWN_AS    compound → PART_OF
agent → ACTED_BY       poss → BELONGS_TO
```

If both entities share a verbal head (siblings in the parse tree), the verb lemma becomes the relation type.

**Subject-Verb-Object (SVO) Triple Extraction**
A separate pass identifies every `VERB` token, collects its `nsubj`/`nsubjpass` children as subjects and `dobj`/`attr`/`pobj` children (including through prepositional phrases) as objects. Each subject-object pair forms a triple with the verb lemma as relation.

Compound expansion ensures multi-word names are captured: for each subject/object token, the system collects `compound`, `amod`, and `flat` children and concatenates them in document order. "Sam Altman" isn't split into two separate entities.

Common verbs are normalized to semantic relation types: `FOUND→FOUNDED_BY`, `DEVELOP→DEVELOPED`, `BASE→BASED_IN`, etc.

**Pronominal Coreference Resolution**
A lightweight heuristic maps pronouns to their most likely antecedent without running a full coreference model:

- `he/she/him/her/his` → nearest preceding `PERSON` entity
- `it/its` → nearest preceding `ORG/PRODUCT/GPE` entity  
- `they/them/their` → nearest preceding entity of any type

This resolves ~70-80% of pronominal references at zero cost. Relations extracted with pronouns as source/target get rewritten to the resolved entity name.

**Noun Phrase Entity Extraction**
spaCy's noun chunker catches compound concepts that NER misses. Filtered to only keep phrases with adjective/noun modifiers (e.g., "Constitutional AI", "protein folding", "reinforcement learning") and excluding pronouns, determiners, and single-word chunks without modifiers. These are typed as `Concept` with lower confidence (0.6).

### Phase 2: Gemini Augmentation (API, Selective)

The LLM receives the chunk text plus a list of already-extracted entities and relations. Its prompt explicitly says: "find ADDITIONAL semantic relationships the NLP missed. Do NOT repeat what's already extracted."

This targets what spaCy genuinely can't do: causal chains ("X led to Y"), implicit associations ("the company, which later became..."), temporal sequences, and part-of hierarchies that aren't syntactically marked.

The augmentation is optional and configurable:
- `gemini_extraction_mode="augment"` — NLP first, Gemini fills gaps (default)
- `gemini_extraction_mode="full"` — Gemini does everything (higher quality, higher cost)

---

## 3. Semantic Chunking Strategy

Traditional chunking splits on character count, often mid-sentence. This system chunks on **sentence boundaries** using spaCy's sentence segmentation.

**Parameters:**
- `semantic_chunk_sentences=5` — each chunk contains 5 sentences
- `semantic_chunk_overlap=1` — 1 sentence overlap between consecutive chunks

**Why sentence-boundary chunking matters:**
- No broken sentences → better embedding quality
- Each chunk is a coherent thought unit
- Entity tracking per chunk is accurate (NER runs on complete sentences)

**Entity tracking:** Each chunk records which named entities appear within its character span. This powers the Graph→Vector Bridge — the graph knows which chunks contain which entities.

**Token estimation:** Each chunk stores a `token_estimate` (word count) for downstream cost estimation.

---

## 4. Page Index System

The `PageIndex` provides structural awareness for documents with page breaks, sections, or headings.

**Detection patterns:**
- Form feed characters (`\f`)
- Horizontal rule separators (`---`, `===`)
- Explicit page markers (`Page N`)
- Markdown headings (`# Section`, `## Subsection`)
- Numbered sections (`1. Introduction`)

Each page/section stores: page number, heading text, and character offset range (start_char, end_char).

**Usage in retrieval:** Chunks carry their page info as metadata. This enables page-filtered queries ("What's on page 5?") and section-aware retrieval.

**Persistence:** The page index is stored as a JSON file, surviving across sessions. On incremental re-ingestion, stale page data is removed before new pages are indexed.

---

## 5. Graph Data Model

### Node Types

**Entity** — the primary knowledge unit.
```
(:Entity {
    name: "Sam Altman",           // Canonical name (title case)
    entity_type: "Person",        // Ontology type
    description: "CEO of OpenAI", // Generated by Gemini or extracted
    confidence: 0.9,              // 0.0–1.0, propagated from source
    mentions: 5,                  // Cross-chunk frequency
    source_id: "doc_001",         // Origin document
    extraction_method: "merged",  // "nlp" | "llm" | "merged"
    created_at: datetime,
    updated_at: datetime
})
```

Entities also receive a **dynamic secondary label** matching their type (`:Person`, `:Organization`, `:Location`) for efficient type-filtered queries.

**DocumentMeta** — source tracking for lifecycle management.
```
(:DocumentMeta {
    doc_id, source_name, content_hash, status,
    chunk_count, entity_count, relation_count,
    ingested_at, updated_at
})
```

**Community** — GraphRAG community summaries.
```
(:Community {
    community_id, level, title, summary, rank, entity_count
})
```

**ChunkRef** — lightweight reference node bridging graph and vector store.
```
(:ChunkRef {chunk_id, doc_id})
```

### Edge Types

**Dynamic relation edges** between entities use the extracted relation type as the edge label: `FOUNDED_BY`, `DEVELOPED`, `HEADQUARTERED_IN`, etc. Each carries:
```
{confidence, weight, description, source_id, extraction_method, created_at}
```

**Structural edges:**
- `(:Entity)-[:MENTIONED_IN]->(:ChunkRef)` — graph↔vector bridge
- `(:Entity)-[:BELONGS_TO_COMMUNITY]->(:Community)` — community membership

### Upsert Semantics

Entities use `MERGE` on `name`. On match:
- Higher confidence wins for `entity_type` and `confidence`
- Longer description wins (more informative)
- `mentions` accumulate (sum)
- `extraction_method` becomes `"merged"` if NLP and LLM both contributed

Relations use `MERGE` on `(source_entity)-[TYPE]->(target_entity)`. On match:
- Higher confidence wins
- `weight` accumulates (frequency signal)
- Longer description wins

This means re-ingesting documents enriches existing knowledge rather than duplicating it.

---

## 6. Microsoft GraphRAG: Community Detection & Summarization

The system implements the core pattern from [Microsoft's GraphRAG paper](https://arxiv.org/abs/2404.16130): detect communities of densely connected entities, summarize each community, and use those summaries for global search.

### Community Detection

**Algorithm:** Leiden (preferred) or Louvain (fallback), executed via Neo4j Graph Data Science (GDS) plugin.

**Process:**
1. Project all `Entity` nodes and their relationships into a GDS in-memory graph
2. Run Leiden with configurable resolution (higher = more granular communities)
3. Write community assignments back to entity nodes as `community_level_0`
4. Intermediate communities stored as `community_levels` for hierarchical access
5. Drop the GDS projection

**Why Leiden over Louvain:** Leiden guarantees well-connected communities (no disconnected subclusters), runs faster on large graphs, and produces more stable results across runs.

### Community Summarization

For each community with 2+ members:
1. Retrieve all intra-community entities and their relationships
2. Send entity names + relation descriptions to Gemini
3. Gemini generates a `title` (2-5 words) and `summary` (2-3 sentences)
4. The summary is embedded via Gemini's embedding model
5. Stored as a `Community` node in Neo4j AND in a dedicated ChromaDB collection

### Global Search

When a query is broad/thematic ("What are the major themes in AI research?"), the community channel provides high-level context that individual entity lookups miss.

Two parallel community retrieval paths:
1. **Graph-side:** Communities ranked by `rank` (proportion of total entities they contain)
2. **Vector-side:** Cosine similarity between query embedding and community summary embeddings

---

## 7. The Graph↔Vector Bridge

This is the key architectural innovation that connects the knowledge graph with the vector store bidirectionally.

**The problem:** Graph retrieval returns structured triples ("Sam Altman FOUNDED_BY OpenAI") but lacks the rich textual context around those facts. Vector retrieval returns relevant text chunks but can't follow relationships.

**The bridge:** During ingestion, each chunk's NLP-extracted entities are linked to `ChunkRef` nodes via `MENTIONED_IN` edges:

```
(:Entity {name: "Sam Altman"})-[:MENTIONED_IN]->(:ChunkRef {chunk_id: "doc_001__chunk_3"})
```

**During retrieval (Channel 4):**
1. NLP extracts seed entities from the query
2. Graph lookup: find all `ChunkRef` nodes linked to those entities via `MENTIONED_IN`
3. Vector lookup: fetch those specific chunks from ChromaDB by ID
4. Result: the exact text passages that _contain_ the graph entities

This gives the retriever high-signal text context that's guaranteed to be entity-relevant, without relying on embedding similarity alone.

---

## 8. 4-Channel Hybrid Retrieval

### Channel 1: Graph Local Search

**Trigger:** NLP extracts seed entities from the query.

**Method:** BFS subgraph expansion via APOC's `subgraphAll` starting from fuzzy-matched entity nodes. Configurable hop depth (default 2). Returns relationship triples with descriptions, confidence scores, and source attribution.

Additionally, for the top 3 seed entities, a separate `retrieve_entity_context` call fetches all direct relationships (both incoming and outgoing), providing a complete local neighborhood view.

**Strengths:** Multi-hop reasoning, relationship-aware, precise for factual queries.
**Weaknesses:** Depends on entity extraction quality from the query.

### Channel 2: Vector Semantic Search

**Trigger:** Always active.

**Method:** Embed the query using Gemini's `retrieval_query` task type, then cosine similarity search against the chunk collection in ChromaDB.

**Strengths:** Handles semantic paraphrasing, works even when entities aren't explicitly named.
**Weaknesses:** No structural relationship awareness.

### Channel 3: Community Global Search

**Trigger:** Active by default (`use_communities=True`).

**Method:** Dual path — graph-ranked community summaries (by entity count proportion) + vector-similarity on community summary embeddings.

**Strengths:** Answers broad thematic queries that no single entity or chunk can address.
**Weaknesses:** Lower specificity, useful primarily as supplementary context.

### Channel 4: Graph→Vector Bridge

**Trigger:** Active when seed entities are found (`use_bridge=True`).

**Method:** Graph query finds `ChunkRef` nodes linked to seed entities, then fetches those chunks from ChromaDB by ID.

**Strengths:** High-signal context guaranteed to contain relevant entities. Combines graph precision with vector richness.
**Weaknesses:** Limited to chunks that contain explicitly named entities.

---

## 9. Reciprocal Rank Fusion (RRF)

All four channels produce ranked result lists. RRF merges them into a single ranking.

**Formula:**

```
score(d) = Σ  weight_i / (k + rank_i)
           i ∈ channels where d appears
```

Where `k=60` (standard RRF constant that prevents top-ranked items from dominating).

**Channel weights:**
| Channel | Weight | Rationale |
|---------|--------|-----------|
| Graph Local | 1.2× | Structural relationships are high-signal |
| Graph→Vector Bridge | 1.1× | Entity-linked context is precise |
| Vector Semantic | 1.0× | Baseline semantic relevance |
| Community Global | 0.8× | Broad context, lower specificity |

**Post-fusion adjustments:**
- **Confirmed sources** get a `1.5×` boost on the fused score
- **Deprecated sources** get a `0.5×` penalty

Items appearing in multiple channels naturally get boosted (their RRF scores accumulate), which is exactly the desired behavior — a result found by both graph traversal and vector similarity is almost certainly relevant.

---

## 10. Incremental Ingestion

The system never re-processes unchanged content.

**Hash-based change detection:**
1. On ingest, compute `SHA-256(text)` for the document
2. Compare against stored hash in `doc_hashes.json`
3. Match → skip entirely (return `{"status": "skipped"}`)
4. Mismatch → content changed:
   - Delete old subgraph (relations, orphan entities, chunk refs)
   - Delete old vectors from ChromaDB
   - Remove old page index entries
   - Re-run full extraction pipeline
   - Update stored hash

**Garbage collection via `sync()`:**
Given a set of doc_ids that should exist, `sync()` identifies and removes any docs that are in the hash store but not in the provided set. This handles document deletion from the source.

**Why not diff-based updates?** Diffing at the triple level (add new triples, remove deleted ones, update changed ones) is fragile because extraction is non-deterministic — the same text can produce slightly different triples on different runs. Clean removal + full re-extraction is simpler and guarantees consistency.

---

## 11. Knowledge Lifecycle & Confidence Model

### Confidence Scores

Every entity and relation carries a `confidence` float (0.0–1.0):

| Source | Default Confidence |
|--------|-------------------|
| spaCy NER entities | 0.85 |
| Dependency-parsed relations | 0.75 |
| SVO triple relations | 0.70 |
| Noun phrase entities | 0.60 |
| Gemini-augmented entities | 0.90 |
| Gemini-augmented relations | 0.85 |

### Source Lifecycle

```
  PENDING ──────► CONFIRMED
     │                │
     │                │ (can be reversed)
     ▼                ▼
  DEPRECATED    DEPRECATED
```

- **Pending** — default state for new sources. No confidence modification.
- **Confirmed** — all triples from this source set to confidence 1.0. Retrieved results get `1.5×` score boost.
- **Deprecated** — all triple confidences halved. Retrieved results get `0.5×` score penalty.

### Conflict Detection

`detect_conflicts(entity_name)` finds cases where the same entity has the same relation type pointing to different targets from different sources. For example:

```
Source A: "Anthropic" --[RAISED]--> "$7 Billion"
Source B: "Anthropic" --[RAISED]--> "$10 Billion"
```

This surfaces contradictions for human review rather than silently keeping both.

### Pruning

`prune_low_confidence(threshold)` removes all relations below the threshold. Default is 0.5. This is useful after deprecating sources — their halved confidence values may fall below the threshold, effectively removing their knowledge.

---

## 12. Cost Optimization Strategy

### Embedding Cost

The biggest API cost driver is embedding. The system batches aggressively:
- 100 texts per `embed_content` call (configurable)
- A 1000-chunk corpus requires ~10 API calls instead of 1000
- Community summaries are embedded in the same batch flow

### Extraction Cost

| Without NLP-First | With NLP-First |
|---|---|
| 1 Gemini call per chunk | 0 calls for NLP (local) |
| | 1 call per chunk for augmentation |
| | **But:** augmentation can be skipped for clear-cut chunks |

In `"augment"` mode, Gemini only processes chunks where NLP found entities — chunks with no NLP extractions are unlikely to yield LLM extractions either (they're probably boilerplate/filler text).

### Incremental Cost

Hash-based skipping means re-running the pipeline on an unchanged corpus costs **zero API calls**. Only modified/new documents incur extraction and embedding costs.

### Rate Limiting

Built-in rate limiter ensures `60 / rpm_limit` seconds minimum gap between consecutive Gemini calls. Default `rpm_limit=15` means ~4 seconds between calls, well within free-tier limits.

---

## 13. Failure Modes & Graceful Degradation

| Failure | Degradation |
|---------|-------------|
| Neo4j GDS not installed | Community detection skips; communities empty; 3-channel retrieval still works |
| APOC not installed | Subgraph traversal falls back to manual Cypher; less efficient but functional |
| Gemini API rate limit | Tenacity retries with exponential backoff (3 attempts) |
| Gemini API down | NLP extraction still runs; embeddings fail → chunks stored without vectors |
| spaCy model not found | Falls back to `en_core_web_sm`; if that fails, downloads it automatically |
| Neo4j connection lost | Graph operations fail; vector operations continue independently |
| ChromaDB corruption | Re-ingest (hash store can be cleared to force full re-processing) |
| Empty query entities | Noun phrase fallback for graph seeds; vector search still runs |

The system is designed so that losing any single component degrades quality but doesn't crash the pipeline.
