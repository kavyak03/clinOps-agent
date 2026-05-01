from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


class Embedder(Protocol):
    dim: int

    def embed_text(self, text: str) -> np.ndarray:
        ...


class _MiniLMEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_name)
        self.dim = 384

    def embed_text(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True)
        return np.array(v[0], dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return _MiniLMEmbedder(model)
