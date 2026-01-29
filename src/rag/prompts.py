SYSTEM_PROMPT = """You are a clinical assistant for a synthetic dataset.
You MUST answer using ONLY the EVIDENCE provided.
If evidence is insufficient, say so.
Return ONLY valid JSON matching the schema."""

def build_user_prompt(question: str, evidence_blocks: list[dict]) -> str:
    ev = "\n\n".join(
        [f"[{i}] doc_id={b['doc_id']} title={b.get('title','')}\n{b['chunk']}" for i,b in enumerate(evidence_blocks)]
    )
    schema = """
Schema:
{
  "answer": string,
  "recommendations": [string],
  "citations": [{"claim": string, "evidence_ids": [int]}],
  "uncertainties": [string]
}
"""
    return f"{schema}\nQuestion: {question}\n\nEVIDENCE:\n{ev}"
