"""PostgreSQL + pgvector store. Install: pip install psycopg2-binary pgvector"""
from __future__ import annotations
from cognity_ai.stores.vector.base import BaseVectorStore
from cognity_ai.models.retrieval import RetrievalResult, SemanticChunk, CommunityInfo


class PgVectorStore(BaseVectorStore):
    def __init__(self, pgvector_config):
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2-binary not installed. Run: pip install psycopg2-binary")

        import psycopg2
        dsn = getattr(pgvector_config, "dsn", "postgresql://localhost/raglib")
        self._table = getattr(pgvector_config, "table_name", "raglib_chunks")
        self._comm_table = getattr(pgvector_config, "community_table", "raglib_communities")
        self._dim = getattr(pgvector_config, "dimension", 768)
        self._conn = psycopg2.connect(dsn)
        self._ensure_tables()

    def _ensure_tables(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_num INT DEFAULT 0,
                    entity_names TEXT DEFAULT '',
                    is_parent BOOLEAN DEFAULT FALSE,
                    parent_chunk_id TEXT DEFAULT '',
                    embedding vector({self._dim})
                );
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                    ON {self._table} USING ivfflat (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS {self._table}_doc_id_idx
                    ON {self._table} (doc_id);
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._comm_table} (
                    community_id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    rank FLOAT DEFAULT 0,
                    embedding vector({self._dim})
                );
            """)
            self._conn.commit()

    def upsert_chunks(self, chunks: list[SemanticChunk]):
        with self._conn.cursor() as cur:
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                cur.execute(f"""
                    INSERT INTO {self._table}
                        (chunk_id, doc_id, text, page_num, entity_names,
                         is_parent, parent_chunk_id, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        entity_names = EXCLUDED.entity_names
                """, (
                    chunk.chunk_id, chunk.doc_id, chunk.text,
                    chunk.page_info.page_num if chunk.page_info else 0,
                    ",".join(chunk.entity_names),
                    chunk.is_parent, chunk.parent_chunk_id or "",
                    chunk.embedding,
                ))
        self._conn.commit()

    def query_chunks(self, embedding: list[float], top_k: int = 10,
                     filters: dict = None) -> list[RetrievalResult]:
        with self._conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, doc_id, text, page_num,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self._table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, embedding, top_k))
            rows = cur.fetchall()
        return [
            RetrievalResult(
                content=row[2], score=float(row[4]), source="vector",
                metadata={"chunk_id": row[0], "doc_id": row[1], "page_num": row[3]},
            )
            for row in rows
        ]

    def query_by_chunk_ids(self, chunk_ids: list[str]) -> list[RetrievalResult]:
        if not chunk_ids:
            return []
        placeholders = ",".join(["%s"] * len(chunk_ids))
        with self._conn.cursor() as cur:
            cur.execute(f"""
                SELECT chunk_id, doc_id, text FROM {self._table}
                WHERE chunk_id IN ({placeholders})
            """, chunk_ids)
            rows = cur.fetchall()
        return [
            RetrievalResult(
                content=row[2], score=1.0, source="vector_bridge",
                metadata={"chunk_id": row[0], "doc_id": row[1]},
            )
            for row in rows
        ]

    def upsert_community(self, community: CommunityInfo):
        if community.embedding is None:
            return
        summary_text = f"[{community.title}] {community.summary}"
        with self._conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self._comm_table} (community_id, title, summary, rank, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (community_id) DO UPDATE SET
                    summary = EXCLUDED.summary, embedding = EXCLUDED.embedding
            """, (community.community_id, community.title, community.summary,
                  community.rank, community.embedding))
        self._conn.commit()

    def query_communities(self, embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        with self._conn.cursor() as cur:
            cur.execute(f"""
                SELECT community_id, title, summary,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self._comm_table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, embedding, top_k))
            rows = cur.fetchall()
        return [
            RetrievalResult(
                content=f"[{row[1]}] {row[2]}",
                score=float(row[3]), source="community",
                metadata={"community_id": row[0]},
            )
            for row in rows
        ]

    def delete_by_doc_id(self, doc_id: str):
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE doc_id = %s", (doc_id,))
        self._conn.commit()
