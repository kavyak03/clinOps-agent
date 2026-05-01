from __future__ import annotations
from typing import Dict, List

def apply_simple_intervention(cohort: List[Dict[str, object]], intervention: str = "improved_glycemic_control") -> List[Dict[str, object]]:
    out = []
    for row in cohort:
        r = dict(row)
        if intervention == "improved_glycemic_control" and isinstance(r.get("a1c"), (int, float)):
            r["a1c"] = round(max(5.5, float(r["a1c"]) - 0.6), 2)
            if isinstance(r.get("outcome_risk"), (int, float)):
                r["outcome_risk"] = round(max(0.01, float(r["outcome_risk"]) - 0.05), 3)
        out.append(r)
    return out
