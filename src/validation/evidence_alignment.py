from __future__ import annotations
from typing import Dict, List

def run_evidence_alignment(candidate_insight: str, evidence: List[Dict[str, object]]) -> Dict[str, object]:
    txt = candidate_insight.lower()
    chunks = " ".join(str(e.get("chunk", "")) for e in evidence).lower()
    keyword_overlap = sum(1 for token in ("egfr", "metformin", "ckd", "a1c", "risk") if token in txt and token in chunks)
    grounded = keyword_overlap >= 2 or len(evidence) > 0
    contradiction = ("no evidence" in txt and len(evidence) > 0)
    return {
        "evidence_count": len(evidence),
        "keyword_overlap": keyword_overlap,
        "grounded": grounded and not contradiction,
        "contradiction_detected": contradiction,
        "passed": bool(grounded and not contradiction)
    }
