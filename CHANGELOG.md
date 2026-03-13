# Changelog

All notable changes to HybridGraphRAG will be documented in this file.

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
