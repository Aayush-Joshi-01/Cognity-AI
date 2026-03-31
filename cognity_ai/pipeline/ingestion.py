"""
IngestionPipeline — plugin-aware refactor of hybrid_rag/ingestion.py.

Uses plugin interfaces for all components so any combination of
embedder / graph store / vector store / extractor / chunker / page index works.
"""
from __future__ import annotations

from cognity_ai.models.knowledge import Entity, Relation, ExtractionResult
from cognity_ai.models.retrieval import SemanticChunk, CommunityInfo
from cognity_ai.models.document import Document
from cognity_ai.extractors.base import BaseExtractor
from cognity_ai.chunkers.base import BaseChunker
from cognity_ai.embedders.base import BaseEmbedder
from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.stores.graph.base import BaseGraphStore
from cognity_ai.page_index.base import BasePageIndex
from cognity_ai.utils.hash import HashStore, content_hash


class IngestionPipeline:
    def __init__(
        self,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore | None,
        page_index: BasePageIndex,
        hash_store: HashStore,
        generator=None,   # optional, used for community summarization
        config=None,
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.page_index = page_index
        self.hash_store = hash_store
        self.generator = generator
        self.config = config

    # ══════════════════════════════════════════════════════════════════════
    # PRIMARY INGESTION
    # ══════════════════════════════════════════════════════════════════════

    def ingest(
        self,
        doc_id: str,
        text: str,
        source_name: str = "",
        status: str = "pending",
        loader_metadata: dict | None = None,
    ) -> dict:
        """
        Ingest a single document given as raw text.

        Steps:
          1. Hash check — skip if unchanged
          2. Page/section detection
          3. Chunking
          4. Knowledge extraction per chunk (NLP + optional LLM augmentation)
          5. Cross-chunk deduplication
          6. Batch embedding
          7. Graph upsert (entities, relations, chunk-entity links)
          8. Vector upsert
          9. Hash store update
        """
        doc_hash = content_hash(text)

        # ── 1. Incremental check ────────────────────────────────────────
        if self.hash_store.is_unchanged(doc_id, text):
            return {"doc_id": doc_id, "status": "skipped", "reason": "unchanged"}

        # Clear stale data if doc already existed
        if self.hash_store.get(doc_id):
            if self.graph_store:
                try:
                    self.graph_store.remove_doc_subgraph(doc_id)
                except Exception:
                    pass
            try:
                self.vector_store.delete_by_doc_id(doc_id)
            except Exception:
                pass
            self.page_index.remove(doc_id)

        # ── 2. Page/section detection ───────────────────────────────────
        pages = self.page_index.detect_pages(text, doc_id, loader_metadata)
        self.page_index.persist()

        # ── 3. Chunking ─────────────────────────────────────────────────
        chunks: list[SemanticChunk] = self.chunker.chunk(text, doc_id, pages)

        # ── 4. Extraction per chunk ─────────────────────────────────────
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []

        for chunk in chunks:
            try:
                result = self.extractor.extract(chunk.text, source_id=doc_id)
            except Exception as e:
                print(f"  Extraction failed for {chunk.chunk_id}: {e}")
                result = ExtractionResult()

            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
            chunk.entity_names = [e.name for e in result.entities]

        # ── 5. Cross-chunk deduplication ────────────────────────────────
        entity_map: dict[str, Entity] = {}
        for ent in all_entities:
            key = ent.name.lower().strip()
            if key in entity_map:
                existing = entity_map[key]
                if ent.confidence > existing.confidence:
                    ent.mentions += existing.mentions
                    entity_map[key] = ent
                else:
                    existing.mentions += ent.mentions
            else:
                entity_map[key] = ent
        unique_entities = list(entity_map.values())

        rel_map: dict[str, Relation] = {}
        for rel in all_relations:
            key = f"{rel.source_entity.lower()}|{rel.relation_type}|{rel.target_entity.lower()}"
            if key in rel_map:
                existing = rel_map[key]
                existing.weight += rel.weight
                if rel.confidence > existing.confidence:
                    rel_map[key] = rel
                    rel_map[key].weight = existing.weight
            else:
                rel_map[key] = rel
        unique_relations = list(rel_map.values())

        # ── 6. Batch embedding ──────────────────────────────────────────
        texts_to_embed = [c.text for c in chunks]
        try:
            embeddings = self.embedder.embed_batch(texts_to_embed)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
        except Exception as e:
            print(f"  Embedding failed: {e}")

        # ── 7. Graph upsert ─────────────────────────────────────────────
        if self.graph_store:
            for ent in unique_entities:
                try:
                    self.graph_store.upsert_entity(ent)
                except Exception:
                    pass
            for rel in unique_relations:
                try:
                    self.graph_store.upsert_relation(rel)
                except Exception:
                    pass
            for chunk in chunks:
                if chunk.entity_names:
                    try:
                        self.graph_store.link_chunk_to_entities(
                            chunk.chunk_id, doc_id, chunk.entity_names
                        )
                    except Exception:
                        pass

            stats = {
                "chunks": len(chunks),
                "entities": len(unique_entities),
                "relations": len(unique_relations),
            }
            try:
                self.graph_store.upsert_doc_meta(doc_id, doc_hash, source_name, status, stats)
            except Exception:
                pass

        # ── 8. Vector upsert ────────────────────────────────────────────
        try:
            self.vector_store.upsert_chunks([c for c in chunks if c.embedding is not None])
        except Exception as e:
            print(f"  Vector upsert failed: {e}")

        # ── 9. Hash update ──────────────────────────────────────────────
        self.hash_store.set(doc_id, doc_hash)

        return {
            "doc_id": doc_id, "status": "ingested",
            "pages": len(pages),
            "chunks": len(chunks),
            "entities": len(unique_entities),
            "relations": len(unique_relations),
        }

    def ingest_document(self, document: Document, status: str = "pending") -> dict:
        """Ingest from a Document object (produced by loaders)."""
        loader_metadata = document.page_map or None
        return self.ingest(
            doc_id=document.doc_id,
            text=document.text,
            source_name=document.source_name or document.source_path,
            status=status,
            loader_metadata=loader_metadata,
        )

    def ingest_batch(self, documents: list[dict]) -> list[dict]:
        """Ingest multiple raw-text documents."""
        results = []
        for doc in documents:
            r = self.ingest(
                doc_id=doc["doc_id"],
                text=doc["text"],
                source_name=doc.get("source_name", ""),
                status=doc.get("status", "pending"),
            )
            results.append(r)
            print(
                f"  [{r['status'].upper():>8}] {doc['doc_id']} — "
                f"{r.get('entities', 0)} entities, {r.get('relations', 0)} relations"
            )
        return results

    def ingest_document_batch(self, documents: list[Document],
                               status: str = "pending") -> list[dict]:
        """Ingest multiple Document objects (from loaders)."""
        results = []
        for document in documents:
            r = self.ingest_document(document, status=status)
            results.append(r)
            print(
                f"  [{r['status'].upper():>8}] {document.doc_id} — "
                f"{r.get('entities', 0)} entities, {r.get('relations', 0)} relations"
            )
        return results

    # ══════════════════════════════════════════════════════════════════════
    # COMMUNITY BUILDING
    # ══════════════════════════════════════════════════════════════════════

    def build_communities(self) -> list[CommunityInfo]:
        """
        Post-ingestion: detect communities, summarize each with LLM, embed + store.
        Requires graph_store to support detect_communities().
        """
        if not self.graph_store:
            print("  No graph store — skipping community detection.")
            return []

        print("  Detecting communities...")
        raw_communities = self.graph_store.detect_communities()
        communities: list[CommunityInfo] = []

        for raw in raw_communities:
            cid = str(raw.get("community_id", raw.get("id", "0")))
            members = raw.get("members", [])
            if len(members) < 2:
                continue

            details = self.graph_store.get_community_entities(raw.get("community_id", 0))
            rel_descs = []
            for d in details:
                if d.get("rel") and d.get("target"):
                    rel_descs.append(
                        f"{d.get('entity')} --[{d['rel']}]--> {d['target']} {d.get('rel_desc', '')}"
                    )

            # Summarize
            summary_data = self._summarize_community(members, rel_descs, cid)

            # Embed summary
            summary_text = f"{summary_data['title']}: {summary_data['summary']}"
            try:
                embedding = self.embedder.embed_batch([summary_text])[0]
            except Exception:
                embedding = None

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

            self.graph_store.store_community_summary(community)
            if embedding:
                self.vector_store.upsert_community(community)

            print(f"  Community [{cid}] '{community.title}': {len(members)} entities")

        return communities

    def _summarize_community(self, members: list[str], rel_descs: list[str],
                              community_id: str) -> dict:
        """Summarize community using LLM generator if available."""
        if self.generator:
            try:
                prompt = (
                    f"You are summarizing a knowledge graph community.\n\n"
                    f"Entities: {', '.join(members[:20])}\n"
                    f"Relationships:\n" + "\n".join(rel_descs[:15]) +
                    "\n\nRespond with JSON: {\"title\": \"<2-5 word title>\", "
                    "\"summary\": \"<2-3 sentence summary>\"}"
                )
                response = self.generator.generate("", prompt)
                import json, re
                # Strip markdown fences
                clean = re.sub(r"```(?:json)?", "", response).strip().strip("`")
                data = json.loads(clean)
                return {"title": data.get("title", f"Community {community_id}"),
                        "summary": data.get("summary", ", ".join(members[:5]))}
            except Exception:
                pass
        return {
            "title": f"Community {community_id}",
            "summary": f"Group of {len(members)} related entities: {', '.join(members[:5])}",
        }

    # ══════════════════════════════════════════════════════════════════════
    # MAINTENANCE
    # ══════════════════════════════════════════════════════════════════════

    def remove_document(self, doc_id: str):
        if self.graph_store:
            try:
                self.graph_store.remove_doc_subgraph(doc_id)
            except Exception:
                pass
        self.vector_store.delete_by_doc_id(doc_id)
        self.page_index.remove(doc_id)
        self.hash_store.remove(doc_id)

    def sync(self, current_doc_ids: set[str]) -> list[str]:
        """Remove documents that are no longer in the given set."""
        stale = self.hash_store.all_doc_ids() - current_doc_ids
        for doc_id in stale:
            print(f"  Removing stale: {doc_id}")
            self.remove_document(doc_id)
        return list(stale)
