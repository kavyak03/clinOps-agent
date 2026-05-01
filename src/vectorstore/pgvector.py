from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from sqlalchemy import create_engine, text

from src.embeddings.embedder import get_embedder


def _to_pgvector_literal(vec: np.ndarray) -> str:
    """
    Convert numpy vector to pgvector literal format.

    pgvector expects a string like:
      '[0.1,0.2,0.3]'

    Without this conversion, SQLAlchemy/psycopg2 may send Python lists
    as numeric[], which breaks pgvector similarity operators like <=>.
    """
    arr = vec.astype(float).tolist()
    return "[" + ",".join(str(x) for x in arr) + "]"


class PgVectorStore:
    """
    pgvector-backed vector store.

    Expected schema:
      embeddings(
        doc_id TEXT,
        title TEXT,
        chunk TEXT,
        url TEXT,
        embedding VECTOR(384)
      )

    Notes:
      - Query and document embeddings must come from the same embedding model.
      - The embedding dimension must match the database vector dimension.
    """

    def __init__(self) -> None:
        self.database_url = os.environ["DATABASE_URL"]
        self.table = os.getenv("PGVECTOR_TABLE", "embeddings")
        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.embedder = get_embedder()

    def search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        """
        Search for top-k semantically similar chunks.

        Local/demo note:
        For very small corpora, approximate IVFFlat indexes can behave oddly
        and may return zero rows for some queries. To make local testing
        reliable, this method disables index scans inside the transaction,
        forcing an exact scan over the small embeddings table.
        """
        q_emb = self.embedder.embed_text(query)
        q_vec = _to_pgvector_literal(q_emb)

        sql = text(
            f"""
            SELECT doc_id, title, chunk, url,
                   1 - (embedding <=> CAST(:q AS vector)) AS score
            FROM {self.table}
            ORDER BY embedding <=> CAST(:q AS vector)
            LIMIT :k;
            """
        )

        with self.engine.begin() as conn:
            # Force exact search for small local/demo corpora.
            # Approximate vector indexes are useful at scale, but for tiny
            # local datasets they can return unstable or empty results.
            conn.execute(text("SET LOCAL enable_indexscan = off"))
            conn.execute(text("SET LOCAL enable_bitmapscan = off"))

            rows = conn.execute(sql, {"q": q_vec, "k": k}).mappings().all()

        return [
            {
                "doc_id": r["doc_id"],
                "title": r.get("title"),
                "chunk": r["chunk"],
                "url": r.get("url"),
                "score": float(r["score"]) if r.get("score") is not None else None,
            }
            for r in rows
        ]

    def upsert_texts(self, texts: List[Dict[str, str]]) -> int:
        """
        Insert document chunks and their embeddings into pgvector.

        Expected input:
          [
            {
              "doc_id": "...",
              "title": "...",
              "chunk": "...",
              "url": "..."
            }
          ]
        """
        if not texts:
            return 0

        sql = text(
            f"""
            INSERT INTO {self.table} (doc_id, title, chunk, url, embedding)
            VALUES (:doc_id, :title, :chunk, :url, CAST(:embedding AS vector))
            """
        )

        with self.engine.begin() as conn:
            for t in texts:
                emb = self.embedder.embed_text(t["chunk"])
                emb_vec = _to_pgvector_literal(emb)

                conn.execute(
                    sql,
                    {
                        "doc_id": t["doc_id"],
                        "title": t.get("title"),
                        "chunk": t["chunk"],
                        "url": t.get("url"),
                        "embedding": emb_vec,
                    },
                )

        return len(texts)