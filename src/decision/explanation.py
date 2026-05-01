from __future__ import annotations
from typing import Dict

def build_explanation(candidate_insight: str, validation_summary: Dict[str, object]) -> str:
    status = validation_summary.get("validation_status", "unknown")
    failed = validation_summary.get("failed_checks", [])
    if status == "validated":
        return "The candidate insight was accepted because statistical, biological, cohort, and evidence alignment checks passed."
    if status == "weakly_supported":
        return f"The candidate insight is only partially supported. Failed checks: {', '.join(failed) if failed else 'none specified'}."
    return f"The candidate insight was rejected because too many validation gates failed: {', '.join(failed) if failed else 'unspecified'}."
