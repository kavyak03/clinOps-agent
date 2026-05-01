from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import requests


def has_citations(answer_obj: Dict[str, Any]) -> bool:
    c = answer_obj.get("citations") or []
    return isinstance(c, list) and len(c) > 0


def run_eval(api_base: str, path: str, endpoint: str = "/agent/ask") -> Dict[str, Any]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    total = len(items)
    citations_ok = 0
    refusal_ok = 0
    latencies: List[int] = []

    for it in items:
        q = it["question"]
        expected_refusal = bool(it.get("expected_refusal", False))

        t0 = time.time()
        r = requests.post(f"{api_base}{endpoint}", json={"question": q, "k": 5}, timeout=60)
        ms = int((time.time() - t0) * 1000)
        latencies.append(ms)

        r.raise_for_status()
        obj = r.json()

        refused = obj.get("refusal_reason") is not None
        if expected_refusal == refused:
            refusal_ok += 1

        if has_citations(obj) or refused:
            citations_ok += 1

    return {
        "total": total,
        "citations_or_refusal_rate": citations_ok / max(total, 1),
        "refusal_accuracy": refusal_ok / max(total, 1),
        "avg_latency_ms": sum(latencies) / max(len(latencies), 1),
        "endpoint": endpoint,
    }


if __name__ == "__main__":
    api = os.getenv("EVAL_API_BASE", "http://localhost:8000")
    data = os.getenv("EVAL_SET", "eval/clinops_eval.jsonl")
    res = run_eval(api, data, endpoint="/agent/ask")
    print(json.dumps(res, indent=2))
