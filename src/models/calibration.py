from __future__ import annotations
from typing import Dict, List

def simple_calibration_summary(scored_cohort: List[Dict[str, object]]) -> Dict[str, float]:
    pred = [float(x.get("predicted_risk", 0.0)) for x in scored_cohort]
    obs = [float(x.get("outcome_risk", 0.0)) for x in scored_cohort]
    if not pred:
        return {"mean_predicted": 0.0, "mean_observed": 0.0, "gap": 0.0}
    mean_pred = sum(pred) / len(pred)
    mean_obs = sum(obs) / len(obs)
    return {"mean_predicted": round(mean_pred, 4), "mean_observed": round(mean_obs, 4), "gap": round(mean_pred - mean_obs, 4)}
