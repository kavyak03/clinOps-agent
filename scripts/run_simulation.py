from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from src.simulation.cohort_generator import generate_synthetic_cohort
from src.simulation.noise_injection import inject_noise
from src.simulation.bias_injection import inject_bias

def main():
    cohort = generate_synthetic_cohort(n=40)
    cohort = inject_noise(cohort, noise_scale=0.03)
    cohort = inject_bias(cohort, subgroup="F", egfr_shift=-3.0)
    print(json.dumps({"n": len(cohort), "example": cohort[0]}, indent=2))

if __name__ == "__main__":
    main()
