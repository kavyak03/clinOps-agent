import re
from typing import Dict, List

def strict_claim_support(answer_json: Dict, evidence_blocks: List[Dict]) -> Dict:
    """Heuristic-but-stronger grounding check for RAG answers.

    Returns:
      - citation_validity_rate: citations that point to valid evidence_ids
      - supported_claim_rate: citations whose claim is supported by cited evidence
      - avg_support_score: average overlap score among supported citations
      - unsupported_claims: list of unsupported citations with reasons
    """
    citations = answer_json.get("citations", []) or []
    if not citations:
        return {
            "citation_validity_rate": 1.0,
            "supported_claim_rate": 1.0,
            "avg_support_score": 1.0,
            "n_citations": 0,
            "unsupported_claims": [],
        }

    n = len(citations)
    valid = 0
    supported = 0
    scores = []
    unsupported_details = []

    for c in citations:
        claim = c.get("claim","")
        eids = c.get("evidence_ids", []) or []
        if not all(isinstance(i, int) for i in eids):
            eids = []

        if eids and all(0 <= i < len(evidence_blocks) for i in eids):
            valid += 1
            cited_text = " ".join([_norm(evidence_blocks[i].get("chunk","")) for i in eids])
        else:
            cited_text = ""

        if not cited_text:
            unsupported_details.append({"claim": claim, "reason": "invalid_or_missing_evidence_ids", "evidence_ids": eids})
            continue

        score, reason = _support_score(claim, cited_text)
        if score >= 0.25:
            supported += 1
            scores.append(score)
        else:
            unsupported_details.append({"claim": claim, "reason": reason, "evidence_ids": eids, "support_score": score})

    return {
        "citation_validity_rate": float(valid / max(1, n)),
        "supported_claim_rate": float(supported / max(1, n)),
        "avg_support_score": float(sum(scores)/max(1,len(scores)) if scores else 0.0),
        "n_citations": int(n),
        "unsupported_claims": unsupported_details,
    }

def _support_score(claim: str, evidence: str):
    claim_n = _norm(claim)
    ev_n = _norm(evidence)

    kws = _keywords(claim_n)
    if not kws:
        return 1.0, "no_keywords"

    nums = re.findall(r"\b\d+(?:\.\d+)?\b", claim_n)
    has_num_support = any(x in ev_n for x in nums) if nums else False

    kw_hits = sum(1 for k in kws if k in ev_n)
    score = kw_hits / len(kws)

    if has_num_support:
        score = max(score, 0.35)

    if score >= 0.25:
        return score, "supported"
    if has_num_support:
        return score, "numbers_supported_but_low_keyword_overlap"
    return score, "low_overlap"

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _keywords(s: str):
    tokens = re.findall(r"[a-z0-9]+", s)
    stop = set([
        "the","and","or","to","of","in","a","an","is","are","for","with","on","as","be","by",
        "if","at","from","this","that","it","into","when","should","may","might","can","consider"
    ])
    tokens = [t for t in tokens if len(t) >= 4 and t not in stop]
    uniq = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
    return uniq[:50]
