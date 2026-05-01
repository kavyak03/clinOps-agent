from __future__ import annotations
import random
from typing import Dict, List

def inject_noise(cohort: List[Dict[str, object]], noise_scale: float = 0.05, seed: int = 7) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    out = []
    for row in cohort:
        r = dict(row)
        for key in ("egfr", "creatinine", "a1c", "ldl"):
            if key in r and isinstance(r[key], (int, float)):
                val = float(r[key])
                r[key] = round(val * (1 + rng.uniform(-noise_scale, noise_scale)), 3)
        out.append(r)
    return out
