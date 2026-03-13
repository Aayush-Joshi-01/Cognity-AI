"""
Ingestion Pipeline — The core data insertion engine.

Flow per document:
  1. Hash check → skip if unchanged (incremental)
  2. Page/section detection → page index
  3. Semantic chunking (sentence-boundary-aware via spaCy)
  4. NLP extraction per chunk (local, free): NER + SVO + dependency triples
  5. [Optional] Gemini augmentation (semantic relations NLP misses)
  6. Entity deduplication + mention counting across chunks
  7. Batch embed all chunks (single Gemini call)
  8. Upsert: graph (entities + relations + chunk-entity links) + vectors
  9. Community detection + summarization (periodic, not per-doc)

Cost optimization:
  - spaCy handles ~70% of extraction at zero API cost
  - Gemini only augments what NLP misses
  - Batch embeddings (100 texts per call)
  - Hash-based skip for unchanged docs
  - Rate limiting built into Gemini calls
"""

import hashlib
import json
from pathlib import Path
from collections import defaultdict

from config import Config
from models import Entity, Relation, ExtractionResult, SemanticChunk, CommunityInfo
from nlp_processor import NLPProcessor
from gemini_extractor import GeminiExtractor
from graph_manager import GraphManager
from vector_manager import VectorManager
from page_index import PageIndex


class IngestionPipeline:
    def __init__(self, nlp: NLPProcessor, gemini: GeminiExtractor,
                 graph: GraphManager, vector: VectorManager,
                 page_idx: PageIndex, config: Config):
        self.nlp = nlp
        self.gemini = gemini
        self.graph = graph
        self.vector = vector
        self.page_idx = page_idx
        self.config = config
        self.hash_path = Path(config.ingestion.hash_store_path)
        self._hashes = self._load_hashes()

    def _load_hashes(self) -> dict[str, str]:
        if self.hash_path.exists():
            return json.loads(self.hash_path.read_text())
        return {}

    def _save_hashes(self):
        self.hash_path.write_text(json.dumps(self._hashes, indent=2))

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ══════════════════════════════════════════════════════════════════════
    # MAIN INGESTION
    # ══════════════════════════════════════════════════════════════════════

    def ingest(self, doc_id: str, text: str, source_name: str = "",
               status: str = "pending") -> dict:
        content_hash = self._content_hash(text)

        # ── Incremental check ───────────────────────────────────────────
        if self._hashes.get(doc_id) == content_hash:
            return {"doc_id": doc_id, "status": "skipped", "reason": "unchanged"}

        # Content changed → clear stale data
        if doc_id in self._hashes:
            self.graph.remove_doc_subgraph(doc_id)
            self.vector.delete_by_doc(doc_id)
            self.page_idx.remove(doc_id)

        # ── 1. Page/Section detection ───────────────────────────────────
        pages = self.nlp.detect_pages(text)
        self.page_idx.store(doc_id, pages)

        # ── 2. Semantic chunking ────────────────────────────────────────
        chunks: list[SemanticChunk] = self.nlp.semantic_chunk(text, doc_id, pages)

        # ── 3. NLP extraction per chunk ─────────────────────────────────
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        chunk_entity_map: dict[str, list[str]] = {}  # chunk_id → entity names

        for chunk in chunks:
            nlp_result = self.nlp.process(chunk.text, source_id=doc_id)

            # ── 4. Gemini augmentation (if enabled) ─────────────────────
            if (self.config.ingestion.use_local_nlp_first
                    and self.config.ingestion.gemini_extraction_mode == "augment"):
                try:
                    llm_result = self.gemini.augment_extraction(
                        chunk.text, nlp_result, source_id=doc_id
                    )
                    nlp_result.entities.extend(llm_result.entities)
                    nlp_result.relations.extend(llm_result.relations)
                except Exception as e:
                    print(f"  Gemini augment failed for {chunk.chunk_id}: {e}")

            elif self.config.ingestion.gemini_extraction_mode == "full":
                # Full Gemini extraction (skip NLP)
                try:
                    llm_result = self.gemini.augment_extraction(
                        chunk.text, ExtractionResult(), source_id=doc_id
                    )
                    nlp_result = llm_result
                except Exception as e:
                    print(f"  Gemini full extraction failed: {e}")

            all_entities.extend(nlp_result.entities)
            all_relations.extend(nlp_result.relations)

            # Track entity→chunk mapping
            ent_names = [e.name for e in nlp_result.entities]
            chunk.entity_names = ent_names
            chunk_entity_map[chunk.chunk_id] = ent_names

        # ── 5. Global deduplication ─────────────────────────────────────
        entity_map: dict[str, Entity] = {}
        for ent in all_entities:
            key = ent.name.lower().strip()
            if key in entity_map:
                existing = entity_map[key]
                # Merge: keep higher confidence, accumulate mentions
                if ent.confidence > existing.confidence:
                    ent.mentions += existing.mentions
                    entity_map[key] = ent
                else:
                    existing.mentions += ent.mentions
            else:
                entity_map[key] = ent
        unique_entities = list(entity_map.values())

        # Deduplicate relations
        rel_map: dict[str, Relation] = {}
        for rel in all_relations:
            key = f"{rel.source_entity.lower()}|{rel.relation_type}|{rel.target_entity.lower()}"
            if key in rel_map:
                existing = rel_map[key]
                existing.weight += rel.weight
                if rel.confidence > existing.confidence:
                    rel_map[key] = rel
                    rel_map[key].weight = existing.weight + rel.weight
            else:
                rel_map[key] = rel
        unique_relations = list(rel_map.values())

        # ── 6. Batch embed chunks ───────────────────────────────────────
        texts_to_embed = [c.text for c in chunks]
        embeddings = self.gemini.embed_batch(texts_to_embed)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # ── 7. Upsert to graph ──────────────────────────────────────────
        for ent in unique_entities:
            self.graph.upsert_entity(ent)
        for rel in unique_relations:
            self.graph.upsert_relation(rel)

        # Chunk-entity links (graph↔vector bridge)
        for chunk in chunks:
            if chunk.entity_names:
                self.graph.link_chunk_to_entities(
                    chunk.chunk_id, doc_id, chunk.entity_names
                )

        # Doc meta
        stats = {
            "chunks": len(chunks),
            "entities": len(unique_entities),
            "relations": len(unique_relations),
        }
        self.graph.upsert_doc_meta(doc_id, content_hash, source_name, status, stats)

        # ── 8. Upsert to vector store ───────────────────────────────────
        self.vector.upsert_chunks(chunks)

        # ── 9. Update hash ──────────────────────────────────────────────
        self._hashes[doc_id] = content_hash
        self._save_hashes()

        return {
            "doc_id": doc_id, "status": "ingested",
            "pages": len(pages), **stats,
        }

    def ingest_batch(self, documents: list[dict]) -> list[dict]:
        results = []
        for doc in documents:
            r = self.ingest(
                doc_id=doc["doc_id"], text=doc["text"],
                source_name=doc.get("source_name", ""),
                status=doc.get("status", "pending"),
            )
            results.append(r)
            print(f"  [{r['status'].upper():>8}] {doc['doc_id']} — "
                  f"{r.get('entities', 0)} entities, {r.get('relations', 0)} relations")
        return results

    # ══════════════════════════════════════════════════════════════════════
    # COMMUNITY BUILDING (Microsoft GraphRAG)
    # ══════════════════════════════════════════════════════════════════════

    def build_communities(self) -> list[CommunityInfo]:
        """
        Post-ingestion step: detect communities via Leiden, summarize each,
        embed summaries, store in both graph and vector.
        """
        print("  Detecting communities via Leiden...")
        raw_communities = self.graph.detect_communities()

        communities = []
        for raw in raw_communities:
            cid = str(raw["community_id"])
            members = raw["members"]
            if len(members) < 2:
                continue

            # Get intra-community relations for context
            details = self.graph.get_community_entities(raw["community_id"])
            rel_descs = []
            for d in details:
                if d["rel"] and d["target"]:
                    desc = d["rel_desc"] or ""
                    rel_descs.append(f"{d['entity']} --[{d['rel']}]--> {d['target']} {desc}")

            # Summarize via Gemini
            try:
                summary_data = self.gemini.summarize_community(members, rel_descs)
            except Exception as e:
                print(f"  Community {cid} summarization failed: {e}")
                summary_data = {"title": f"Community {cid}", "summary": ", ".join(members[:5])}

            # Embed the summary
            summary_text = f"{summary_data['title']}: {summary_data['summary']}"
            embedding = self.gemini.embed_batch([summary_text])[0]

            community = CommunityInfo(
                community_id=cid,
                level=0,
                entity_names=members,
                title=summary_data["title"],
                summary=summary_data["summary"],
                rank=len(members) / max(len(raw_communities), 1),
                embedding=embedding,
            )
            communities.append(community)

            # Store in graph and vector
            self.graph.store_community_summary(community)
            self.vector.upsert_community(community)

            print(f"  Community [{cid}] '{community.title}': {len(members)} entities")

        return communities

    # ══════════════════════════════════════════════════════════════════════
    # MAINTENANCE
    # ══════════════════════════════════════════════════════════════════════

    def remove_document(self, doc_id: str):
        self.graph.remove_doc_subgraph(doc_id)
        self.vector.delete_by_doc(doc_id)
        self.page_idx.remove(doc_id)
        self._hashes.pop(doc_id, None)
        self._save_hashes()

    def sync(self, current_doc_ids: set[str]) -> list[str]:
        stale = set(self._hashes.keys()) - current_doc_ids
        for doc_id in stale:
            print(f"  Removing stale: {doc_id}")
            self.remove_document(doc_id)
        return list(stale)
