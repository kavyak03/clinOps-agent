from __future__ import annotations

from typing import Any, Dict, List, Union

try:
    # LangChain Document type (optional dependency)
    from langchain_core.documents import Document
except Exception:
    Document = None  # type: ignore


def _doc_to_evidence(d: Any, i: int) -> Dict[str, Any]:
    """Normalize either a dict-like evidence or a LangChain Document into a dict evidence record."""
    # Case 1: already a dict
    if isinstance(d, dict):
        ev = dict(d)
        ev.setdefault("evidence_id", ev.get("doc_id", f"doc_{i}"))
        ev.setdefault("title", ev.get("title", ev.get("doc_id", f"doc_{i}")))
        ev.setdefault("text", ev.get("text") or ev.get("chunk") or ev.get("content") or "")
        return ev

    # Case 2: LangChain Document
    if Document is not None and isinstance(d, Document):
        meta = d.metadata or {}
        return {
            "evidence_id": meta.get("evidence_id") or meta.get("doc_id") or f"doc_{i}",
            "title": meta.get("title") or meta.get("doc_id") or f"doc_{i}",
            "text": d.page_content,
            "metadata": meta,
        }

    # Case 3: fallback
    return {
        "evidence_id": f"doc_{i}",
        "title": f"doc_{i}",
        "text": str(d),
    }


def generate_answer_offline_stub(question: str, evidence: List[Any]) -> Dict[str, Any]:
    # Normalize evidence
    ev = [_doc_to_evidence(e, i) for i, e in enumerate(evidence)]

    # Minimal, deterministic offline response
    # (Keep this aligned with your repo’s JSON schema if you have one)
    answer = {
        "question": question,
        "answer": "OFFLINE_STUB: This is a placeholder answer. Use llm='openai' for model-generated answers.",
        "citations": [e["evidence_id"] for e in ev[:3]],
        "evidence": ev[:5],
    }
    return answer

def generate_answer_openai(question: str, evidence, model: str = "gpt-4o-mini"):
    """
    Optional OpenAI generator.
    - Keeps the import available so prototypes don't crash.
    - If openai deps/key aren't present, raises a clear error.
    """
    try:
        import os
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "OpenAI generator requested but OpenAI dependencies are not installed. "
            "Install requirements_openai.txt and try again."
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Normalize evidence into a readable string for the prompt
    ev_texts = []
    for i, d in enumerate(evidence):
        if isinstance(d, dict):
            title = d.get("title") or d.get("doc_id") or f"doc_{i}"
            text = d.get("text") or ""
        else:
            # LangChain Document fallback
            title = getattr(d, "metadata", {}) or {}
            title = title.get("title") or title.get("doc_id") or f"doc_{i}"
            text = getattr(d, "page_content", str(d))
        ev_texts.append(f"[{i+1}] {title}\n{text}")

    prompt = f"""You are a careful clinical assistant.
Answer the question using ONLY the evidence below.
Return JSON with keys: answer, citations.

Question: {question}

Evidence:
{'\n\n'.join(ev_texts)}
"""

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # Return raw text; your caller can json.loads if desired
    return {"raw": resp.choices[0].message.content}
