from __future__ import annotations
from typing import Dict, List

def run_cohort_consistency_checks(cohort: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(cohort)
    missing_core = 0
    sex_counts = {}
    for row in cohort:
        for k in ("egfr", "a1c", "creatinine"):
            if row.get(k) in (None, ""):
                missing_core += 1
        sx = row.get("sex", "UNK")
        sex_counts[sx] = sex_counts.get(sx, 0) + 1
    subgroup_imbalance = False
    if n:
        max_prop = max(sex_counts.values()) / n
        subgroup_imbalance = max_prop > 0.85
    return {
        "n_samples": n,
        "missing_core_fields": missing_core,
        "subgroup_counts": sex_counts,
        "subgroup_imbalance": subgroup_imbalance,
        "passed": bool(n >= 20 and missing_core <= max(3, int(n * 0.1)))
    }
