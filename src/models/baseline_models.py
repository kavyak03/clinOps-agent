from __future__ import annotations
from typing import Dict, List
from .risk_model import score_cohort

def run_baseline_models(cohort: List[Dict[str, object]]) -> Dict[str, object]:
    scored = score_cohort(cohort)
    mean_risk = round(sum(float(r["predicted_risk"]) for r in scored) / max(len(scored), 1), 4)
    return {"scored_cohort": scored, "summary": {"n": len(scored), "mean_predicted_risk": mean_risk}}
