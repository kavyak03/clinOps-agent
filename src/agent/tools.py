from __future__ import annotations

import re
from typing import Any, Dict, List


def extract_citations(_: Dict[str, Any], evidence: List[Dict[str, object]]) -> Dict[str, Any]:
    pmids = set()
    years = set()
    for e in evidence:
        chunk = str(e.get("chunk") or "")
        for m in re.findall(r"\bPMID:\s*(\d{5,10})\b", chunk):
            pmids.add(m)
        for y in re.findall(r"\b(19\d{2}|20\d{2})\b", chunk):
            years.add(y)
    return {"pmids": sorted(pmids)[:15], "years": sorted(years)[:10]}


def grade_evidence(params: Dict[str, Any], evidence: List[Dict[str, object]]) -> Dict[str, Any]:
    graded = []
    for e in evidence:
        text = (str(e.get("chunk") or "")).lower()
        score = 0
        if "randomized" in text or "controlled trial" in text or "trial" in text:
            score += 2
        if "meta-analysis" in text or "systematic review" in text:
            score += 1
        if "case report" in text or ("n=" in text and "n=1" in text):
            score -= 1
        graded.append({"doc_id": e.get("doc_id"), "score": score})
    graded.sort(key=lambda x: x["score"], reverse=True)
    return {"graded": graded[:10], "method": "heuristic"}


TOOLS = {
    "extract_citations": extract_citations,
    "grade_evidence": grade_evidence,
}
