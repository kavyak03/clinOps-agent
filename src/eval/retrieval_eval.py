import json
from pathlib import Path
from typing import List, Dict, Tuple

def recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    topk = retrieved_doc_ids[:k]
    return 1.0 if any(d in set(gold_doc_ids) for d in topk) else 0.0

def mrr_at_k(retrieved_doc_ids: List[str], gold_doc_ids: List[str], k: int) -> float:
    gold = set(gold_doc_ids)
    for rank, d in enumerate(retrieved_doc_ids[:k], start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0

def evaluate_questions(questions: List[Dict], retrieve_fn, ks=(1,3,5,10)) -> Dict:
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"mrr@{k}": 0.0 for k in ks})
    n = 0
    per_q = []

    for q in questions:
        text = q["question"]
        gold = q["gold_doc_ids"]
        hits = retrieve_fn(text)
        retrieved = [h["doc_id"] for h in hits]
        row = {"question_id": q.get("question_id"), "question": text, "gold_doc_ids": gold, "retrieved_doc_ids": retrieved}
        for k in ks:
            r = recall_at_k(retrieved, gold, k)
            m = mrr_at_k(retrieved, gold, k)
            totals[f"recall@{k}"] += r
            totals[f"mrr@{k}"] += m
            row[f"recall@{k}"] = r
            row[f"mrr@{k}"] = m
        per_q.append(row)
        n += 1

    if n == 0:
        raise ValueError("No questions provided.")

    summary = {k: v / n for k, v in totals.items()}
    summary["n_questions"] = n
    return {"summary": summary, "per_question": per_q}

def load_questions_jsonl(path: Path) -> List[Dict]:
    qs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))
    return qs
