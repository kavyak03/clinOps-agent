from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore


class CrossEncoderReranker:
    """
    Two-stage retrieval helper:
      1) broad vector retrieval (higher recall)
      2) cross-encoder reranking (higher precision)

    Environment:
      RERANK_ENABLE=true|false
      RERANK_CANDIDATES=20
      RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("RERANK_ENABLE", "true").lower().strip() == "true"
        self.candidate_k = int(os.getenv("RERANK_CANDIDATES", "20"))
        self.model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if CrossEncoder is None:
                raise RuntimeError("sentence-transformers CrossEncoder is not available.")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, evidence: List[Dict[str, object]], top_k: int) -> List[Dict[str, object]]:
        if not evidence:
            return []
        if not self.enabled:
            return evidence[:top_k]

        pairs = [(query, str(item.get("chunk") or "")) for item in evidence]
        try:
            scores = self.model.predict(pairs)
        except Exception:
            # Safe fallback: preserve original vector-ranked order
            return evidence[:top_k]

        reranked: List[Dict[str, object]] = []
        for item, score in zip(evidence, scores):
            enriched = dict(item)
            enriched["vector_score"] = enriched.get("score")
            enriched["rerank_score"] = float(score)
            enriched["score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda x: float(x.get("rerank_score", -1e9)), reverse=True)
        return reranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
