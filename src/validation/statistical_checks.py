from __future__ import annotations
from typing import Dict, List
from src.models.calibration import simple_calibration_summary

def run_statistical_checks(scored_cohort: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(scored_cohort)
    calib = simple_calibration_summary(scored_cohort)
    enough_samples = n >= 20
    direction_ok = calib["mean_predicted"] >= 0 and calib["mean_observed"] >= 0
    return {
        "n_samples": n,
        "enough_samples": enough_samples,
        "calibration": calib,
        "direction_ok": direction_ok,
        "passed": bool(enough_samples and abs(calib["gap"]) <= 0.25 and direction_ok)
    }
