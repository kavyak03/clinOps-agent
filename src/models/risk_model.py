from __future__ import annotations
from typing import Dict, List

def score_patient(row: Dict[str, object]) -> float:
    egfr = float(row.get("egfr", 60))
    a1c = float(row.get("a1c", 6.5))
    creat = float(row.get("creatinine", 1.0))
    age = float(row.get("age", 50))
    risk = 0.05
    if egfr < 45: risk += 0.20
    if egfr < 30: risk += 0.15
    if a1c > 7.0: risk += 0.10
    if creat > 1.5: risk += 0.07
    if age > 65: risk += 0.08
    return round(min(0.99, max(0.01, risk)), 3)

def score_cohort(cohort: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in cohort:
        r = dict(row)
        r["predicted_risk"] = score_patient(r)
        out.append(r)
    return out
