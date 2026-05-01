from __future__ import annotations

from typing import Any, Dict, List


class OfflineStubLLM:
    provider_name = "offline"
    model_name = "offline_stub_v1"

    def plan(self, question: str) -> Dict[str, Any]:
        return {
            "query": question,
            "tools": [
                {"name": "grade_evidence", "input": {"method": "heuristic"}},
                {"name": "extract_citations", "input": {"mode": "simple"}},
            ],
        }

    def answer_with_citations(self, question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = evidence[:3]
        if not top:
            return {
                "answer": "I couldn't find relevant evidence in the indexed corpus.",
                "recommendations": [],
                "citations": [],
                "uncertainties": ["No evidence retrieved."],
            }

        citations = []
        bullets = []
        for i, e in enumerate(top, start=1):
            title = e.get("title") or e.get("doc_id") or f"doc_{i}"
            chunk = (e.get("chunk") or "").strip().replace("\n", " ")
            citations.append({"ref": f"[{i}]", "title": str(title), "doc_id": str(e.get("doc_id")), "url": e.get("url")})
            bullets.append(f"- ({i}) {chunk[:240]}...")

        answer = (
            "Offline stub answer (no external LLM). Based on top retrieved evidence:\n"
            + "\n".join(bullets)
            + "\n\nSet LLM_PROVIDER=anthropic or openai for real generations."
        )
        return {"answer": answer, "citations": citations, "uncertainties": ["Offline stub; not a real generation."]}

    def synthesize(self, question: str, evidence: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = self.answer_with_citations(question, evidence)
        base["uncertainties"] = base.get("uncertainties", []) + ["Heuristic tool results used; verify clinically."]
        return base
