from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os
import requests

from src.simulation.cohort_generator import generate_synthetic_cohort
from src.models.baseline_models import run_baseline_models
from src.validation.statistical_checks import run_statistical_checks
from src.validation.cohort_consistency import run_cohort_consistency_checks
from src.validation.biological_checks import run_biological_checks
from src.validation.evidence_alignment import run_evidence_alignment
from src.validation.decision_gate import make_validation_decision
from src.decision.formatter import format_final_output
from src.decision.recommendation import make_recommendation
from src.decision.explanation import build_explanation


def derive_candidate_question(cohort_summary: dict) -> str:
    return (
        f"Summarize the relationship between eGFR, A1c, and CKD-related risk for a cohort "
        f"with mean predicted risk {cohort_summary.get('mean_predicted_risk', 'unknown')}."
    )


def call_rag(api_base: str, question: str, use_agent: bool = False) -> dict:
    endpoint = "/agent/ask" if use_agent else "/ask"
    r = requests.post(
        api_base.rstrip("/") + endpoint,
        json={"question": question, "k": 5},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()


def offline_demo_rag(question: str) -> dict:
    """No-Docker fallback for testing the decision pipeline without API/Postgres."""
    evidence = [
        {
            "doc_id": "offline-egfr",
            "title": "Offline demo eGFR evidence",
            "chunk": "Lower eGFR is associated with worse kidney function and higher CKD progression risk.",
            "score": 1.0,
        },
        {
            "doc_id": "offline-a1c",
            "title": "Offline demo A1c evidence",
            "chunk": "Higher A1c reflects poorer glycemic control and may increase diabetic kidney disease risk.",
            "score": 0.9,
        },
    ]
    return {
        "answer": (
            "Lower eGFR and higher A1c are associated with higher CKD-related risk in this cohort. "
            "This offline demo response is deterministic and does not call the RAG API."
        ),
        "evidence": evidence,
        "citations": [{"id": "1", "doc_id": "offline-egfr"}, {"id": "2", "doc_id": "offline-a1c"}],
        "uncertainties": ["Offline demo mode; no live retrieval or LLM provider was called."],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=os.getenv("PIPELINE_API_BASE", "http://localhost:8000"))
    ap.add_argument("--use-agent", action="store_true")
    ap.add_argument("--offline-demo", action="store_true", help="Run without API/Postgres/LLM using deterministic demo evidence.")
    ap.add_argument("--output", default="reports/decision_output.json")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    cohort = generate_synthetic_cohort(n=args.n)
    model_out = run_baseline_models(cohort)
    question = derive_candidate_question(model_out["summary"])

    if args.offline_demo:
        rag = offline_demo_rag(question)
    else:
        rag = call_rag(args.api_base, question, use_agent=args.use_agent)

    candidate_insight = rag.get("answer", "")
    evidence = rag.get("evidence", [])

    statistical = run_statistical_checks(model_out["scored_cohort"])
    biological = run_biological_checks(candidate_insight)
    cohort_checks = run_cohort_consistency_checks(cohort)
    evidence_checks = run_evidence_alignment(candidate_insight, evidence)
    decision = make_validation_decision(statistical, biological, cohort_checks, evidence_checks)

    final_output = format_final_output(candidate_insight, decision, evidence, statistical, biological, cohort_checks)
    final_output["final_decision"] = make_recommendation(final_output)
    final_output["explanation"] = build_explanation(candidate_insight, decision)
    final_output["question_used"] = question
    final_output["offline_demo"] = bool(args.offline_demo)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    main()
