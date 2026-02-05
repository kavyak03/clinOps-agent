from __future__ import annotations

import json
import os
from typing import Any, Dict, List

def generate_answer_offline_stub(question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic, offline-safe generator.

    Returns a structured JSON answer object without calling any external LLM.
    Useful for reproducibility, CI, and environments where external calls are disallowed.
    """
    top = evidence[:3] if evidence else []
    if not top:
        return {
            "answer": "I couldn't find relevant evidence in the indexed corpus.",
            "recommendations": [],
            "citations": [],
            "uncertainties": ["No evidence retrieved."],
        }

    bullets = []
    citations = []
    recs = []
    for i, e in enumerate(top):
        title = e.get("title") or e.get("doc_id", f"doc_{i}")
        chunk = (e.get("chunk") or "").strip().replace("\n", " ")
        snippet = chunk[:240] + ("..." if len(chunk) > 240 else "")
        bullets.append(f"- {title}: {snippet}")
        recs.append(title)
        citations.append({"claim": title, "evidence_ids": [i]})

    return {
        "answer": (
            "Offline stub answer (no external LLM). Top supporting snippets:\n"
            + "\n".join(bullets)
        ),
        "recommendations": recs,
        "citations": citations,
        "uncertainties": ["LLM disabled in offline demo; plug in an LLM provider for fluent answers."],
    }


def generate_answer_openai(question: str, evidence: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """OpenAI-backed generator.

    Requires:
      - pip install openai
      - OPENAI_API_KEY set in environment

    Note: API usage is separate from ChatGPT Plus.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "PowerShell:  $env:OPENAI_API_KEY='your_key_here'\n"
            "macOS/Linux: export OPENAI_API_KEY='your_key_here'"
        )

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "OpenAI python package not installed. Install with: pip install openai\n"
            f"Original error: {e}"
        )

    # Keep prompt compact
    formatted = []
    for i, e in enumerate(evidence[:8] if evidence else []):
        formatted.append(
            {
                "evidence_id": i,
                "doc_id": e.get("doc_id"),
                "title": e.get("title"),
                "source": e.get("source"),
                "chunk": (e.get("chunk") or "")[:1200],
            }
        )

    system = (
        "You are a careful clinical assistant. Use ONLY the provided evidence. "
        "If evidence is insufficient, say so. Do not invent facts.\n\n"
        "Return STRICT JSON with keys: answer (string), recommendations (list[string]), "
        "citations (list of {claim: string, evidence_ids: list[int]}), uncertainties (list[string]).\n"
        "evidence_ids must refer to the evidence_id values provided."
    )

    user_payload = {"question": question, "evidence": formatted}

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    parsed = _safe_json_parse(content)
    _validate_answer_schema(parsed)
    return parsed


def generate_answer(question: str, evidence: List[Dict[str, Any]], provider: str = "offline", **kwargs: Any) -> Dict[str, Any]:
    """Unified entrypoint for generation.

    provider:
      - "offline": deterministic stub (default)
      - "openai": OpenAI API call (requires billing/quota)
    """
    provider = (provider or "offline").lower()
    if provider == "openai":
        return generate_answer_openai(question, evidence, model=kwargs.get("model", "gpt-4o-mini"))
    return generate_answer_offline_stub(question, evidence)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise RuntimeError("LLM did not return valid JSON. First 1200 chars:\n" + text[:1200])


def _validate_answer_schema(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise RuntimeError("LLM output is not a JSON object.")
    for k in ["answer", "recommendations", "citations", "uncertainties"]:
        if k not in obj:
            raise RuntimeError(f"LLM output missing key: {k}")
    if not isinstance(obj["answer"], str):
        raise RuntimeError("answer must be a string")
    if not isinstance(obj["recommendations"], list):
        raise RuntimeError("recommendations must be a list")
    if not isinstance(obj["citations"], list):
        raise RuntimeError("citations must be a list")
    if not isinstance(obj["uncertainties"], list):
        raise RuntimeError("uncertainties must be a list")