from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def generate_answer_offline_stub(question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic offline generator used for fully reproducible runs without any paid API.
    It creates a small grounded summary from the top evidence chunks and attaches evidence_ids.
    """
    top = evidence[:3]
    if not top:
        return {
            "answer": "I couldn't find relevant evidence in the indexed corpus.",
            "recommendations": [],
            "citations": [],
        }

    bullets = []
    citations = []
    recommendations = []

    for i, e in enumerate(top):
        title = e.get("title") or e.get("doc_id", f"doc_{i}")
        chunk = (e.get("chunk") or "").strip().replace("\n", " ")
        snippet = chunk[:220] + ("..." if len(chunk) > 220 else "")
        bullets.append(f"- {title}: {snippet}")
        recommendations.append(title)
        citations.append(
            {
                "claim": title,
                "evidence_ids": [i],  # IMPORTANT: indices refer to the provided evidence list
            }
        )

    answer = (
        "Offline stub answer (replace with LLM for better phrasing). "
        "Here are the most relevant evidence snippets:\n"
        + "\n".join(bullets)
    )

    return {
        "answer": answer,
        "recommendations": recommendations,
        "citations": citations,
    }


def generate_answer_openai(question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    OpenAI-backed generator. Keeps the same signature the CLI expects:
        generate_answer_openai(question, evidence)

    Requires environment variable OPENAI_API_KEY.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment and retry.\n"
            "PowerShell:  $env:OPENAI_API_KEY='your_key_here'\n"
            "macOS/Linux: export OPENAI_API_KEY='your_key_here'"
        )

    # Lazy import so offline mode doesn't require openai package
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "OpenAI python package not installed in this environment.\n"
            "Install with: pip install openai\n"
            f"Original error: {e}"
        )

    # Prepare evidence text (keep it compact)
    formatted_evidence = []
    for i, e in enumerate(evidence[:8]):
        formatted_evidence.append(
            {
                "evidence_id": i,
                "doc_id": e.get("doc_id"),
                "title": e.get("title"),
                "source": e.get("source"),
                "chunk": (e.get("chunk") or "")[:1200],  # avoid huge prompts
            }
        )

    system = (
        "You are a careful clinical assistant. You MUST answer using ONLY the provided evidence. "
        "If evidence is insufficient, say so. Do not invent facts.\n\n"
        "Return STRICT JSON with keys:\n"
        "- answer: string\n"
        "- recommendations: list of short strings\n"
        "- citations: list of objects with keys {claim: string, evidence_ids: list[int]}\n\n"
        "Important: evidence_ids must refer to the evidence_id fields provided."
    )

    user = {
        "question": question,
        "evidence": formatted_evidence,
        "output_format": {
            "answer": "string",
            "recommendations": ["string", "..."],
            "citations": [{"claim": "string", "evidence_ids": [0]}],
        },
    }

    client = OpenAI(api_key=api_key)

    # Model choice: pick something generally available; change if you prefer.
    # If your account has different model availability, swap the model name accordingly.
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content

    # Try to parse JSON. If the model returns extra text, salvage the JSON block.
    parsed = _safe_json_parse(content)
    _validate_answer_schema(parsed)
    return parsed


def _safe_json_parse(text: str) -> Dict[str, Any]:
    text = text.strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # salvage first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise RuntimeError(
        "Model did not return valid JSON. Raw response:\n" + text[:2000]
    )


def _validate_answer_schema(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise RuntimeError("LLM output is not a JSON object.")

    for k in ["answer", "recommendations", "citations"]:
        if k not in obj:
            raise RuntimeError(f"LLM output missing key: {k}")

    if not isinstance(obj["answer"], str):
        raise RuntimeError("answer must be a string")
    if not isinstance(obj["recommendations"], list):
        raise RuntimeError("recommendations must be a list")
    if not isinstance(obj["citations"], list):
        raise RuntimeError("citations must be a list")

    for c in obj["citations"]:
        if not isinstance(c, dict):
            raise RuntimeError("each citation must be an object")
        if "claim" not in c or "evidence_ids" not in c:
            raise RuntimeError("each citation must have claim and evidence_ids")
        if not isinstance(c["claim"], str):
            raise RuntimeError("citation.claim must be a string")
        if not isinstance(c["evidence_ids"], list) or not all(
            isinstance(i, int) for i in c["evidence_ids"]
        ):
            raise RuntimeError("citation.evidence_ids must be a list[int]")