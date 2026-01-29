import json
from pathlib import Path
import faiss

from src.rag.embed import Embedder
from src.rag.retrieve import retrieve
from src.eval.retrieval_eval import load_questions_jsonl, evaluate_questions

def main():
    processed = Path("data/processed")
    meta_path = processed / "guidelines_chunks.json"
    index_path = processed / "guidelines.faiss"
    if not meta_path.exists() or not index_path.exists():
        raise SystemExit("Index not found. Run: python scripts/build_index.py")

    meta = json.loads(meta_path.read_text())
    index = faiss.read_index(str(index_path))
    embedder = Embedder()

    def retrieve_fn(qtext):
        return retrieve(qtext, embedder, index, meta, k=10)

    qs = load_questions_jsonl(Path("data/corpora/questions_gold.jsonl"))
    results = evaluate_questions(qs, retrieve_fn, ks=(1,3,5,10))

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out/"retrieval_eval_summary.json").write_text(json.dumps(results["summary"], indent=2))
    (out/"retrieval_eval_per_question.json").write_text(json.dumps(results["per_question"], indent=2))
    print(json.dumps(results["summary"], indent=2))

if __name__ == "__main__":
    main()
