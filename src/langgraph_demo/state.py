from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class DecisionGraphState(TypedDict, total=False):
    question: str
    cohort: List[Dict[str, Any]]
    model_output: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    candidate_insight: str
    statistical_checks: Dict[str, Any]
    biological_checks: Dict[str, Any]
    cohort_consistency: Dict[str, Any]
    evidence_alignment: Dict[str, Any]
    validation_summary: Dict[str, Any]
    final_output: Dict[str, Any]
    used_langgraph_runtime: bool
