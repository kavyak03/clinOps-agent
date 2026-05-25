from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from src.eval.eval_gate import run_smoke_eval, check_thresholds, load_json
from src.llm.schema import normalize_llm_answer


def main() -> None:
    malformed = {
        "answer": "Demo answer",
        "citations": ["1", {"id": "2"}],
        "uncertainties": "Single uncertainty string",
    }
    normalized = normalize_llm_answer(malformed)

    metrics = run_smoke_eval()
    thresholds = load_json("eval/quality_thresholds.json")
    ok, failures = check_thresholds(metrics, thresholds)

    print(json.dumps({
        "llm_schema_normalization": normalized.model_dump(),
        "smoke_metrics": metrics,
        "thresholds_passed": ok,
        "failures": failures,
    }, indent=2))

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
