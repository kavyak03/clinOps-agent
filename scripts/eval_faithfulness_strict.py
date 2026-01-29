import json
from pathlib import Path
import faiss

from src.rag.embed import Embedder
from src.rag.retrieve import retrieve
from src.rag.generate import generate_answer_offline_stub
from src.eval.faithfulness_strict import strict_claim_support

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
        strict = strict_claim_support(ans, hits)
        rows.append({"question_id": q.get("question_id"), "question": question, **strict})

    out = Path("reports"); out.mkdir(exist_ok=True)
    (out/"faithfulness_strict.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "n_questions": len(rows),
        "avg_supported_claim_rate": sum(r["supported_claim_rate"] for r in rows)/max(1,len(rows)),
        "avg_citation_validity_rate": sum(r["citation_validity_rate"] for r in rows)/max(1,len(rows)),
        "avg_support_score": sum(r["avg_support_score"] for r in rows)/max(1,len(rows)),
    }
    (out/"faithfulness_strict_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
