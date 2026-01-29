import re
from typing import Dict, List

def citation_coverage(answer_json: Dict) -> float:
    recs = answer_json.get("recommendations", []) or []
    cits = answer_json.get("citations", []) or []
    if not recs:
        return 1.0
    covered = 0
    for r in recs:
        r_norm = _norm(r)
        ok = False
        for c in cits:
            claim = _norm(c.get("claim",""))
            if r_norm and (r_norm in claim or claim in r_norm):
                if c.get("evidence_ids"):
                    ok = True
                    break
        covered += 1 if ok else 0
    return covered / max(1, len(recs))

def simple_faithfulness(answer_json: Dict, evidence_blocks: List[Dict]) -> float:
    ans_text = " ".join([answer_json.get("answer","")] + (answer_json.get("recommendations",[]) or []))
    ans_tokens = _keywords(ans_text)
    if not ans_tokens:
        return 1.0
    ev_text = " ".join([b.get("chunk","") for b in evidence_blocks])
    ev_text_n = _norm(ev_text)
    hits = sum(1 for t in ans_tokens if t in ev_text_n)
    return hits / len(ans_tokens)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def _keywords(s: str):
    s = _norm(s)
    tokens = re.findall(r"[a-z0-9]+", s)
    stop = set(["the","and","or","to","of","in","a","an","is","are","for","with","on","as","be","by","if","at","from","this","that","it"])
    tokens = [t for t in tokens if len(t) >= 4 and t not in stop]
    uniq = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)
    return uniq[:40]
