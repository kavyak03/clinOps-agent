from __future__ import annotations
import random
from typing import Dict, List

def generate_synthetic_cohort(n: int = 50, seed: int = 42) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    cohort = []
    for i in range(n):
        egfr = max(12.0, min(95.0, rng.gauss(42, 12)))
        a1c = max(5.2, min(12.0, rng.gauss(7.4, 1.1)))
        creatinine = max(0.5, min(3.5, rng.gauss(1.4, 0.4)))
        age = int(max(30, min(85, rng.gauss(61, 10))))
        sex = "F" if i % 2 == 0 else "M"
        risk = min(0.99, max(0.01, 0.15 + (1 if egfr < 30 else 0) * 0.25 + max(0, a1c - 6.5) * 0.05))
        cohort.append({
            "patient_id": f"SYN_{i:04d}",
            "age": age,
            "sex": sex,
            "egfr": round(egfr, 1),
            "creatinine": round(creatinine, 2),
            "a1c": round(a1c, 2),
            "ldl": int(max(50, min(220, rng.gauss(110, 25)))),
            "outcome_risk": round(risk, 3),
            "cohort": "synthetic_ckd_diabetes"
        })
    return cohort
