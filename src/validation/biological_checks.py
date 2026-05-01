from __future__ import annotations
from typing import Dict

_RULES = [
    {"name": "low_egfr_higher_ckd_risk", "keyword": "egfr"},
    {"name": "high_a1c_higher_metabolic_risk", "keyword": "a1c"},
]

def run_biological_checks(candidate_insight: str) -> Dict[str, object]:
    txt = candidate_insight.lower()
    matched_rules = [r["name"] for r in _RULES if r["keyword"] in txt]
    plausible = True
    contradiction_flags = []
    if "higher egfr increases risk" in txt:
        plausible = False
        contradiction_flags.append("egfr_direction_contradiction")
    if "higher a1c lowers risk" in txt:
        plausible = False
        contradiction_flags.append("a1c_direction_contradiction")
    return {
        "matched_rules": matched_rules,
        "plausible": plausible,
        "contradiction_flags": contradiction_flags,
        "passed": plausible
    }
