import argparse
import json
from pathlib import Path

from src.config import Config
from src.rag.embed import Embedder
from src.rag.retrieve import retrieve
from src.rag.generate import generate_answer_openai, generate_answer_offline_stub

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config()
    processed = Path("data/processed")
    meta = json.loads((processed / "guidelines_chunks.json").read_text())
    import faiss
    index = faiss.read_index(str(processed / "guidelines.faiss"))

    embedder = Embedder(cfg.embed_model_name)
    hits = retrieve(args.question, embedder, index, meta, k=args.top_k)

    print("\n=== Top retrieved evidence ===")
    for i, h in enumerate(hits):
        print(f"\n[{i}] score={h['score']:.3f} doc_id={h['doc_id']} title={h.get('title','')}")
        print(h["chunk"])

    print("\n=== Generated answer (offline stub) ===")
    ans = generate_answer_offline_stub(args.question, hits)
    print(json.dumps(ans, indent=2))

if __name__ == "__main__":
    main()
