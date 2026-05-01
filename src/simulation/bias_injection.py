from __future__ import annotations
from typing import Dict, List

def inject_bias(cohort: List[Dict[str, object]], subgroup: str = "F", egfr_shift: float = -5.0) -> List[Dict[str, object]]:
    out = []
    for row in cohort:
        r = dict(row)
        if r.get("sex") == subgroup and isinstance(r.get("egfr"), (int, float)):
            r["egfr"] = round(float(r["egfr"]) + egfr_shift, 3)
        out.append(r)
    return out
