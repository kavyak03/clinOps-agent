from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
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
    r = requests.post(api_base.rstrip("/") + endpoint, json={"question": question, "k": 5}, timeout=90)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=os.getenv("PIPELINE_API_BASE", "http://localhost:8000"))
    ap.add_argument("--use-agent", action="store_true")
    ap.add_argument("--output", default="reports/decision_output.json")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    cohort = generate_synthetic_cohort(n=args.n)
    model_out = run_baseline_models(cohort)
    question = derive_candidate_question(model_out["summary"])

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

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(json.dumps(final_output, indent=2))

if __name__ == "__main__":
    main()
