from __future__ import annotations

import os

from src.vectorstore.pgvector import PgVectorStore
from src.vectorstore.faiss import FaissStore


def get_vector_store():
    which = os.getenv("VECTOR_STORE", "pgvector").lower().strip()
    if which == "faiss":
        return FaissStore()
    return PgVectorStore()
