import json
from pathlib import Path
import csv

def main():
    rep = Path("reports"); rep.mkdir(exist_ok=True)

    retrieval_per = json.loads((rep/"retrieval_eval_per_question.json").read_text())
    retrieval_sum = json.loads((rep/"retrieval_eval_summary.json").read_text())
    gen_rows = json.loads((rep/"generation_heuristics.json").read_text())
    strict_rows = json.loads((rep/"faithfulness_strict.json").read_text())

    def key(r): return r.get("question_id") or r.get("question")
    gen_map = {key(r): r for r in gen_rows}
    strict_map = {key(r): r for r in strict_rows}

    out_csv = rep/"leaderboard.csv"
    fields = [
        "question_id","question",
        "recall@1","recall@3","recall@5","recall@10",
        "mrr@1","mrr@3","mrr@5","mrr@10",
        "citation_coverage","faithfulness_overlap","n_evidence",
        "citation_validity_rate","supported_claim_rate","avg_support_score","n_citations"
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in retrieval_per:
            k = key(r)
            row = {f: "" for f in fields}
            row["question_id"] = r.get("question_id","")
            row["question"] = r.get("question","")
            for m in ["recall@1","recall@3","recall@5","recall@10","mrr@1","mrr@3","mrr@5","mrr@10"]:
                row[m] = r.get(m,"")

            g = gen_map.get(k, {})
            row["citation_coverage"] = g.get("citation_coverage","")
            row["faithfulness_overlap"] = g.get("faithfulness_overlap","")
            row["n_evidence"] = g.get("n_evidence","")

            s = strict_map.get(k, {})
            row["citation_validity_rate"] = s.get("citation_validity_rate","")
            row["supported_claim_rate"] = s.get("supported_claim_rate","")
            row["avg_support_score"] = s.get("avg_support_score","")
            row["n_citations"] = s.get("n_citations","")
            w.writerow(row)

    (rep/"leaderboard_summary.txt").write_text(
        "=== Retrieval summary ===\n" + json.dumps(retrieval_sum, indent=2) + f"\n\nLeaderboard: {out_csv}\n",
        encoding="utf-8"
    )
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
