"""
FAISS in-memory vector store backend.
Requires: pip install faiss-cpu  (or faiss-gpu)

Uses IndexFlatIP (inner product) with L2-normalised vectors for cosine similarity.
Metadata is kept entirely in Python dicts keyed by sequential integer IDs.
Communities are stored in a separate index with the same approach.

Limitations:
  - No persistence by default (pass index_path to enable save/load).
  - delete_by_doc_id requires a full index rebuild (FAISS has no efficient delete).
"""

import warnings
from raglib.stores.vector.base import BaseVectorStore
from raglib.models.retrieval import SemanticChunk, CommunityInfo, RetrievalResult


class FAISSStore(BaseVectorStore):
    def __init__(self, dimension: int = 768, index_path: str | None = None):
        self._dimension = dimension
        self._index_path = index_path

        # chunk index state
        self._chunks_meta: dict[int, dict] = {}   # sequential int_id → metadata
        self._chunk_id_to_int: dict[str, int] = {}  # chunk_id str → int_id
        self._next_chunk_id = 0
        self._index = None  # lazy faiss.IndexFlatIP

        # community index state
        self._community_meta: dict[int, dict] = {}
        self._community_id_to_int: dict[str, int] = {}
        self._next_community_id = 0
        self._comm_index = None  # lazy faiss.IndexFlatIP

        if index_path:
            self._try_load()

    # ── Internal helpers ────────────────────────────────────────────────

    def _get_chunk_index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self._dimension)
        return self._index

    def _get_comm_index(self):
        if self._comm_index is None:
            import faiss
            self._comm_index = faiss.IndexFlatIP(self._dimension)
        return self._comm_index

    @staticmethod
    def _normalize(vec):
        import numpy as np
        v = np.array(vec, dtype="float32")
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v

    def _try_load(self):
        """Load persisted index + metadata from index_path (if it exists)."""
        import os, pickle, faiss
        base = self._index_path
        chunk_idx_path = base + ".chunks.faiss"
        chunk_meta_path = base + ".chunks.pkl"
        comm_idx_path = base + ".communities.faiss"
        comm_meta_path = base + ".communities.pkl"
        try:
            if os.path.exists(chunk_idx_path):
                self._index = faiss.read_index(chunk_idx_path)
                with open(chunk_meta_path, "rb") as f:
                    state = pickle.load(f)
                    self._chunks_meta = state["meta"]
                    self._chunk_id_to_int = state["id_map"]
                    self._next_chunk_id = state["next_id"]
            if os.path.exists(comm_idx_path):
                self._comm_index = faiss.read_index(comm_idx_path)
                with open(comm_meta_path, "rb") as f:
                    state = pickle.load(f)
                    self._community_meta = state["meta"]
                    self._community_id_to_int = state["id_map"]
                    self._next_community_id = state["next_id"]
        except Exception as e:
            warnings.warn(f"FAISSStore: could not load persisted index: {e}")

    def save(self):
        """Persist the index and metadata to index_path."""
        if not self._index_path:
            return
        import pickle, faiss
        base = self._index_path
        if self._index is not None:
            faiss.write_index(self._index, base + ".chunks.faiss")
            with open(base + ".chunks.pkl", "wb") as f:
                pickle.dump({"meta": self._chunks_meta,
                             "id_map": self._chunk_id_to_int,
                             "next_id": self._next_chunk_id}, f)
        if self._comm_index is not None:
            faiss.write_index(self._comm_index, base + ".communities.faiss")
            with open(base + ".communities.pkl", "wb") as f:
                pickle.dump({"meta": self._community_meta,
                             "id_map": self._community_id_to_int,
                             "next_id": self._next_community_id}, f)

    # ── Chunk Operations ────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        import numpy as np
        index = self._get_chunk_index()
        vecs = []
        for c in chunks:
            if not c.embedding:
                continue
            # If this chunk_id already exists, mark old slot as deleted
            # (FAISS flat index doesn't support true delete; we just add a new
            # vector and update the id map — stale slots are benign for flat search)
            int_id = self._next_chunk_id
            self._chunks_meta[int_id] = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "index": c.index,
                "page_num": c.page_info.page_num if c.page_info else 0,
                "section": c.page_info.section if c.page_info else "",
                "heading": c.page_info.heading if c.page_info else "",
                "entity_names": "|".join(c.entity_names),
                "sentence_count": c.sentence_count,
                "token_estimate": c.token_estimate,
            }
            self._chunk_id_to_int[c.chunk_id] = int_id
            self._next_chunk_id += 1
            vecs.append(self._normalize(c.embedding))

        if vecs:
            mat = np.vstack(vecs)
            index.add(mat)

    def query_chunks(
        self,
        embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        import numpy as np
        index = self._get_chunk_index()
        if index.ntotal == 0:
            return []

        q = self._normalize(embedding).reshape(1, -1)
        # Fetch more candidates when filtering so we can still return top_k after
        fetch_k = min(index.ntotal, top_k * 5 if filters else top_k)
        scores, ids = index.search(q, fetch_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            meta = self._chunks_meta.get(idx)
            if meta is None:
                continue
            # Apply filters
            if filters:
                if "doc_id" in filters and meta.get("doc_id") != filters["doc_id"]:
                    continue
                if "page_num" in filters and meta.get("page_num") != filters["page_num"]:
                    continue
            results.append(
                RetrievalResult(
                    content=meta["text"],
                    score=float(score),
                    source="vector",
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        results = []
        for cid in chunk_ids:
            int_id = self._chunk_id_to_int.get(cid)
            if int_id is None:
                continue
            meta = self._chunks_meta.get(int_id)
            if meta:
                results.append(
                    RetrievalResult(
                        content=meta["text"],
                        score=0.8,
                        source="vector_bridge",
                        metadata=meta,
                    )
                )
        return results

    def delete_by_doc_id(self, doc_id: str):
        """
        FAISS IndexFlatIP has no efficient delete.
        We remove the entries from metadata dicts and rebuild the index.
        """
        import numpy as np, faiss

        # Find int IDs to drop
        drop_ids = {
            iid for iid, m in self._chunks_meta.items() if m.get("doc_id") == doc_id
        }
        if not drop_ids:
            return

        # Collect surviving vectors (we have to re-extract from FAISS)
        keep_ids = [
            iid for iid in sorted(self._chunks_meta)
            if iid not in drop_ids
        ]

        # Reconstruct vectors for survivors
        old_index = self._get_chunk_index()
        if old_index.ntotal > 0 and keep_ids:
            # IndexFlatIP supports reconstruct
            try:
                vecs = np.vstack([
                    old_index.reconstruct(iid) for iid in keep_ids
                ])
            except Exception:
                # Fallback: cannot reconstruct — just clear metadata
                for iid in drop_ids:
                    meta = self._chunks_meta.pop(iid)
                    self._chunk_id_to_int.pop(meta["chunk_id"], None)
                return
        else:
            vecs = None

        # Rebuild index
        new_index = faiss.IndexFlatIP(self._dimension)
        new_meta: dict[int, dict] = {}
        new_id_map: dict[str, int] = {}
        new_next = 0

        if vecs is not None and len(vecs):
            new_index.add(vecs)
            for new_iid, old_iid in enumerate(keep_ids):
                m = self._chunks_meta[old_iid]
                new_meta[new_iid] = m
                new_id_map[m["chunk_id"]] = new_iid
                new_next = new_iid + 1

        self._index = new_index
        self._chunks_meta = new_meta
        self._chunk_id_to_int = new_id_map
        self._next_chunk_id = new_next

    # ── Community Operations ────────────────────────────────────────────

    def upsert_community(self, community: CommunityInfo):
        import numpy as np
        if not community.embedding:
            return
        index = self._get_comm_index()
        int_id = self._next_community_id
        self._community_meta[int_id] = {
            "community_id": community.community_id,
            "text": f"{community.title}: {community.summary}",
            "level": community.level,
            "rank": community.rank,
            "entity_count": len(community.entity_names),
        }
        self._community_id_to_int[community.community_id] = int_id
        self._next_community_id += 1
        v = self._normalize(community.embedding).reshape(1, -1)
        index.add(v)

    def query_communities(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        import numpy as np
        index = self._get_comm_index()
        if index.ntotal == 0:
            return []

        q = self._normalize(embedding).reshape(1, -1)
        k = min(top_k, index.ntotal)
        scores, ids = index.search(q, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            meta = self._community_meta.get(idx)
            if meta:
                results.append(
                    RetrievalResult(
                        content=meta["text"],
                        score=float(score),
                        source="community",
                        metadata=meta,
                    )
                )
        return results

    def count(self) -> dict:
        chunk_idx = self._get_chunk_index()
        comm_idx = self._get_comm_index()
        # ntotal counts all ever-added vectors (including "soft-deleted" ones);
        # use metadata dict length for accurate live counts.
        return {
            "chunks": len(self._chunks_meta),
            "communities": len(self._community_meta),
        }
