import json
from pathlib import Path
import faiss

from src.rag.embed import Embedder
from src.rag.retrieve import retrieve
from src.rag.generate import generate_answer_offline_stub
from src.eval.faithfulness_eval import citation_coverage, simple_faithfulness

def main():
    processed = Path("data/processed")
    meta_path = processed / "guidelines_chunks.json"
    index_path = processed / "guidelines.faiss"
    if not meta_path.exists() or not index_path.exists():
        raise SystemExit("Index not found. Run: python scripts/build_index.py")

    meta = json.loads(meta_path.read_text())
    index = faiss.read_index(str(index_path))
    embedder = Embedder()

    qs = [json.loads(l) for l in Path("data/corpora/questions_gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for q in qs:
        question = q["question"]
        hits = retrieve(question, embedder, index, meta, k=5)
        ans = generate_answer_offline_stub(question, hits)

        rows.append({
            "question_id": q.get("question_id"),
            "question": question,
            "citation_coverage": float(citation_coverage(ans)),
            "faithfulness_overlap": float(simple_faithfulness(ans, hits)),
            "n_evidence": len(hits)
        })

    out = Path("reports"); out.mkdir(exist_ok=True)
    (out/"generation_heuristics.json").write_text(json.dumps(rows, indent=2))
    print("Wrote reports/generation_heuristics.json")

if __name__ == "__main__":
    main()
