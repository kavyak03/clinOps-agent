import argparse
import json
from pathlib import Path

from src.config import Config
from src.rag.chunking import chunk_text
from src.rag.embed import Embedder
from src.rag.index_faiss import build_faiss_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/corpora/guidelines.jsonl",
        help="Path to a JSONL corpus with fields: doc_id, title, source, text",
    )
    args = parser.parse_args()

    cfg = Config()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    chunks = []
    meta = []
    for r in rows:
        text = r["text"]
        for ch in chunk_text(text, max_chars=cfg.chunk_max_chars, overlap=cfg.chunk_overlap):
            chunks.append(ch)
            meta.append({
                "doc_id": r["doc_id"],
                "title": r.get("title",""),
                "source": r.get("source",""),
                "chunk": ch,
            })

    embedder = Embedder(cfg.embed_model_name)
    emb = embedder.encode(chunks)
    index = build_faiss_index(emb)

    import faiss
    faiss.write_index(index, str(out_dir / "guidelines.faiss"))
    (out_dir / "guidelines_chunks.json").write_text(json.dumps(meta, indent=2))

    print(f"Indexed {len(chunks)} chunks from {corpus_path} -> data/processed/guidelines.faiss")

if __name__ == "__main__":
    main()
