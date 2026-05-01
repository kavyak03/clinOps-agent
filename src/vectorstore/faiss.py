from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from src.embeddings.embedder import get_embedder


class FaissStore:
    """Numpy cosine-sim fallback (no faiss dependency)."""

    def __init__(self, path: str = "data/faiss_store.jsonl") -> None:
        self.path = path
        self.embedder = get_embedder()
        self.items: List[Dict[str, object]] = []
        self.mat: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self.items = []
            self.mat = None
            return
        items = []
        embs = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                items.append(obj)
                embs.append(self.embedder.embed_text(obj["chunk"]))
        self.items = items
        self.mat = np.vstack(embs) if embs else None

    def upsert_texts(self, texts: List[Dict[str, str]]) -> int:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps(t) + "\n")
        self._load()
        return len(texts)

    def search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        if self.mat is None or not self.items:
            return []
        q = self.embedder.embed_text(query)
        qn = q / (np.linalg.norm(q) + 1e-9)
        mn = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-9)
        sims = mn @ qn
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            it = dict(self.items[int(i)])
            it["score"] = float(sims[int(i)])
            out.append(it)
        return out
