import json
from pathlib import Path
from .embed import Embedder
from .index_faiss import search

def load_jsonl_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def retrieve(query: str, embedder: Embedder, index, chunk_meta: list[dict], k: int = 5):
    q = embedder.encode([query])
    scores, idxs = search(index, q, k=k)

    results = []
    for score, idx in zip(scores, idxs):
        meta = chunk_meta[int(idx)]
        results.append({
            "score": float(score),
            "doc_id": meta["doc_id"],
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "chunk": meta["chunk"],
        })
    return results
