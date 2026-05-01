from __future__ import annotations
import json
from src.simulation.cohort_generator import generate_synthetic_cohort
from src.models.baseline_models import run_baseline_models
from src.validation.statistical_checks import run_statistical_checks
from src.validation.cohort_consistency import run_cohort_consistency_checks
from src.validation.biological_checks import run_biological_checks
from src.validation.evidence_alignment import run_evidence_alignment
from src.validation.decision_gate import make_validation_decision

def main():
    cohort = generate_synthetic_cohort(40)
    model_out = run_baseline_models(cohort)
    candidate_insight = "Lower eGFR and higher A1c are associated with higher CKD-related risk in this cohort."
    evidence = [{"doc_id": "demo1", "title": "Demo evidence", "chunk": "In CKD populations, lower eGFR is associated with higher progression risk. Higher A1c can worsen metabolic risk.", "score": 0.9}]
    statistical = run_statistical_checks(model_out["scored_cohort"])
    biological = run_biological_checks(candidate_insight)
    cohort_checks = run_cohort_consistency_checks(cohort)
    evidence_checks = run_evidence_alignment(candidate_insight, evidence)
    decision = make_validation_decision(statistical, biological, cohort_checks, evidence_checks)
    print(json.dumps({"statistical": statistical, "biological": biological, "cohort": cohort_checks, "evidence": evidence_checks, "decision": decision}, indent=2))

if __name__ == "__main__":
    main()
