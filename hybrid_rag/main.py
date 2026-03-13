"""
Hybrid Dynamic Graph RAG + Vector RAG — Ultimate Edition

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
  │  Documents   │────►│  NLP Layer   │────►│ Gemini Augment│
  │  (+ pages)   │     │  (spaCy)     │     │ (only gaps)   │
  └─────────────┘     │  • NER       │     └───────┬───────┘
                      │  • SVO       │             │
                      │  • DepParse  │             │
                      │  • Coref     │             │
                      │  • Chunking  │             │
                      └──────┬───────┘             │
                             │          ┌──────────┘
                             ▼          ▼
  ┌──────────────────────────────────────────────────────┐
  │              Ingestion Pipeline                       │
  │  Hash check → Page index → Chunk → Extract → Embed   │
  └──────────────┬────────────────────────┬──────────────┘
                 │                        │
                 ▼                        ▼
  ┌──────────────────┐     ┌──────────────────────┐
  │   Neo4j (Online)  │     │   ChromaDB (Local)    │
  │  • Entities       │     │  • Chunk embeddings   │
  │  • Relations      │     │  • Community embeds   │
  │  • Communities    │     │  • Page metadata      │
  │  • ChunkRef links │◄───►│  • Entity-linked IDs  │
  │  • Doc lifecycle  │     │                        │
  └──────────────────┘     └──────────────────────┘
                 │                        │
                 └────────┬───────────────┘
                          ▼
  ┌──────────────────────────────────────────────────────┐
  │           4-Channel Hybrid Retriever                  │
  │  1. Graph Local (BFS subgraph)                        │
  │  2. Vector Semantic (cosine similarity)               │
  │  3. Community Global (GraphRAG summaries)             │
  │  4. Graph→Vector Bridge (entity-linked chunks)        │
  │                                                       │
  │  → Reciprocal Rank Fusion → Confirmed-source boost    │
  │  → Gemini generation with full context                │
  └──────────────────────────────────────────────────────┘

Cost optimization:
  • spaCy handles ~70% of extraction (free, local)
  • Gemini only augments semantic gaps
  • Batch embeddings (100/call)
  • Hash-based incremental skip
  • Rate limiting on all API calls
"""

from config import Config
from nlp_processor import NLPProcessor
from gemini_extractor import GeminiExtractor
from graph_manager import GraphManager
from vector_manager import VectorManager
from page_index import PageIndex
from ingestion import IngestionPipeline
from hybrid_retriever import HybridRetriever
from knowledge_updater import KnowledgeUpdater


def build_pipeline(config: Config | None = None) -> dict:
    """Wire all components. Returns dict of named components."""
    config = config or Config()

    nlp = NLPProcessor(config.nlp)
    gemini = GeminiExtractor(config.gemini)
    graph = GraphManager(config.neo4j, config.graphrag)
    vector = VectorManager(config.chroma)
    page_idx = PageIndex(config.ingestion.page_index_path)

    pipeline = IngestionPipeline(nlp, gemini, graph, vector, page_idx, config)
    retriever = HybridRetriever(nlp, gemini, graph, vector, config)
    updater = KnowledgeUpdater(graph, config)

    return {
        "config": config,
        "nlp": nlp,
        "gemini": gemini,
        "graph": graph,
        "vector": vector,
        "page_idx": page_idx,
        "pipeline": pipeline,
        "retriever": retriever,
        "updater": updater,
    }


# ════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    c = build_pipeline()
    pipeline = c["pipeline"]
    retriever = c["retriever"]
    updater = c["updater"]
    graph = c["graph"]

    # ── 1. Ingest (incremental, NLP-first) ──────────────────────────────
    documents = [
        {
            "doc_id": "doc_001",
            "text": """
# Overview of OpenAI

OpenAI was founded by Sam Altman, Elon Musk, and others in 2015.
The company developed GPT-4 and DALL-E. Sam Altman serves as CEO.
OpenAI is headquartered in San Francisco, California.
They launched ChatGPT in November 2022 which reached 100 million users in two months.

## Key Products

GPT-4 is a large language model that powers ChatGPT and many third-party applications.
DALL-E generates images from text descriptions using diffusion models.
OpenAI also developed Whisper for speech recognition and Codex for code generation.
Microsoft invested $10 billion in OpenAI in January 2023.
            """,
            "source_name": "tech_overview",
            "status": "confirmed",
        },
        {
            "doc_id": "doc_002",
            "text": """
# Anthropic and AI Safety

Anthropic was founded by Dario Amodei and Daniela Amodei in 2021.
The company focuses on AI safety research and developed the Claude family of models.
Anthropic is based in San Francisco and has raised over $7 billion in funding.
Google and Amazon are major investors in Anthropic.

## Constitutional AI

Anthropic pioneered Constitutional AI (CAI), a technique for training AI systems
to be helpful, harmless, and honest. CAI uses a set of principles to guide model
behavior without extensive human feedback on harmful outputs.
Claude models are trained using RLHF combined with Constitutional AI methods.
            """,
            "source_name": "ai_safety_report",
            "status": "pending",
        },
        {
            "doc_id": "doc_003",
            "text": """
# Google DeepMind

Google DeepMind developed AlphaFold which solved the protein folding problem.
Demis Hassabis leads DeepMind as CEO. They also created the Gemini model family.
DeepMind is a subsidiary of Alphabet and is based in London, United Kingdom.

AlphaGo, developed by DeepMind, defeated world champion Lee Sedol in Go in 2016.
This was considered a landmark achievement in artificial intelligence.
DeepMind's research spans reinforcement learning, neuroscience-inspired AI,
and large language models. They published over 1000 research papers.
            """,
            "source_name": "deepmind_notes",
            "status": "confirmed",
        },
    ]

    print("=" * 70)
    print("STEP 1: Incremental NLP-First Ingestion")
    print("=" * 70)
    results = pipeline.ingest_batch(documents)
    for r in results:
        print(f"  {r}")

    # Re-ingest → should skip all
    print("\n--- Re-ingest (should skip unchanged) ---")
    results = pipeline.ingest_batch(documents)

    # ── 2. Build communities (GraphRAG Leiden) ──────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Community Detection (Microsoft GraphRAG)")
    print("=" * 70)
    try:
        communities = pipeline.build_communities()
        print(f"  Built {len(communities)} communities")
    except Exception as e:
        print(f"  Community detection requires Neo4j GDS plugin: {e}")
        print("  Skipping — communities will be empty for global search")

    # ── 3. Knowledge lifecycle ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Knowledge Lifecycle")
    print("=" * 70)
    updater.confirm_source("doc_002")
    print("  Confirmed doc_002")

    conflicts = updater.detect_conflicts("San Francisco")
    print(f"  Conflicts for 'San Francisco': {len(conflicts)}")
    for cf in conflicts:
        print(f"    {cf}")

    health = updater.health_report()
    print(f"  Health: {health}")

    # ── 4. Hybrid 4-channel queries ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: 4-Channel Hybrid RAG Queries")
    print("=" * 70)

    questions = [
        "Who founded Anthropic and what is Constitutional AI?",
        "What is the relationship between Google, DeepMind, and AI research?",
        "Compare the AI products built by companies in San Francisco.",
        "What breakthroughs has DeepMind achieved?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = retriever.query_with_sources(q)
        print(f"A: {result['answer'][:400]}...")
        print(f"  Seeds: {result['seed_entities']}")
        print(f"  Sources: graph={len(result['sources']['graph'])}, "
              f"vector={len(result['sources']['vector'])}, "
              f"community={len(result['sources']['community'])}")
        print(f"  Top scores:")
        for s in result["retrieval_scores"][:3]:
            print(f"    [{s['channel']:>10}] {s['score']:.4f} — {s['content']}")

    # ── 5. Incremental update (modified doc) ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: Incremental Update")
    print("=" * 70)

    r = pipeline.ingest(
        doc_id="doc_002",
        text="""
# Anthropic — Updated 2025

Anthropic was founded by Dario Amodei and Daniela Amodei in 2021.
The company focuses on AI safety and developed Claude 4, their most capable model.
Anthropic has raised over $10 billion from Google, Amazon, and Salesforce.

## Constitutional AI v2

Anthropic advanced Constitutional AI with scalable oversight techniques.
Claude 4 Opus achieved state-of-the-art performance on reasoning benchmarks.
The company expanded to over 1000 employees across San Francisco and London.
        """,
        source_name="anthropic_2025",
        status="confirmed",
    )
    print(f"  {r}")

    # ── 6. Prune + final health ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Prune & Health")
    print("=" * 70)
    pruned = updater.prune_low_confidence(0.3)
    print(f"  Pruned {pruned} low-confidence relations")

    health = updater.health_report()
    print(f"  Final health: {health}")

    stats = updater.get_source_stats()
    for s in stats:
        print(f"  [{s.get('status', '?'):>10}] {s['doc_id']} — "
              f"{s.get('triples', 0)} triples, {s.get('chunks', 0)} chunks")

    # ── Cleanup ─────────────────────────────────────────────────────────
    graph.close()
    print("\nDone.")
