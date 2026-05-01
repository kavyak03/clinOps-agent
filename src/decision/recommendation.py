from __future__ import annotations
from typing import Dict

def make_recommendation(validation_output: Dict[str, object]) -> str:
    status = validation_output.get("validation_status")
    if status == "validated":
        return "Use as a supported decision-assist summary, with clinician review."
    if status == "weakly_supported":
        return "Treat as hypothesis-generating output; seek more evidence before using operationally."
    return "Do not use this AI-generated insight for decision support without additional evidence."
