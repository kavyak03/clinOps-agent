from __future__ import annotations
from typing import Dict

def make_validation_decision(statistical: Dict[str, object], biological: Dict[str, object], cohort: Dict[str, object], evidence: Dict[str, object]) -> Dict[str, object]:
    passed = {
        "statistical": bool(statistical.get("passed")),
        "biological": bool(biological.get("passed")),
        "cohort": bool(cohort.get("passed")),
        "evidence": bool(evidence.get("passed")),
    }
    n_pass = sum(1 for v in passed.values() if v)
    if n_pass == 4:
        status = "validated"
        confidence = "moderate"
    elif n_pass >= 2:
        status = "weakly_supported"
        confidence = "low"
    else:
        status = "rejected"
        confidence = "low"
    reasons = [k for k, v in passed.items() if not v]
    return {
        "validation_status": status,
        "confidence": confidence,
        "failed_checks": reasons,
        "passed_checks": [k for k, v in passed.items() if v]
    }
