from __future__ import annotations

from typing import Dict, List


def format_final_output(
    candidate_insight: str,
    validation_summary: Dict[str, object],
    supporting_evidence: List[Dict[str, object]],
    statistical: Dict[str, object],
    biological: Dict[str, object],
    cohort: Dict[str, object],
) -> Dict[str, object]:
    return {
        "candidate_insight": candidate_insight,
        "validation_status": validation_summary["validation_status"],
        "confidence": validation_summary["confidence"],
        "supporting_evidence": [
            {
                "doc_id": e.get("doc_id"),
                "title": e.get("title"),
                # After reranking, score is the final ranking score.
                "score": e.get("score"),
                "vector_score": e.get("vector_score"),
                "rerank_score": e.get("rerank_score"),
                "chunk_preview": str(e.get("chunk", ""))[:180],
            }
            for e in supporting_evidence[:5]
        ],
        "statistical_checks": statistical,
        "biological_checks": biological,
        "cohort_consistency": cohort,
    }
