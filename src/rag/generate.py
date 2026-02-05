# src/rag/generate.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _doc_to_evidence(doc: Any, i: int) -> Dict[str, Any]:
    """
    Normalize a retrieved item into a dict evidence object.

    Supports:
      - dict evidence from our core retriever
      - LangChain Document (has .page_content, .metadata)
      - any object with .page_content / .metadata-like attributes
    """
    if isinstance(doc, dict):
        # assume already evidence-like
        ev = dict(doc)
        ev.setdefault("evidence_id", ev.get("id") or ev.get("doc_id") or f"doc_{i}")
        ev.setdefault("text", ev.get("text") or ev.get("content") or "")
        ev.setdefault("title", ev.get("title") or ev.get("doc_id") or ev["evidence_id"])
        return ev

    # LangChain Document or similar
    page_content = getattr(doc, "page_content", None)
    metadata = getattr(doc, "metadata", None) or {}

    if page_content is None and hasattr(doc, "__dict__"):
        # try fallbacks
        page_content = getattr(doc, "content", "") or getattr(doc, "text", "")

    ev_id = metadata.get("evidence_id") or metadata.get("id") or metadata.get("doc_id") or f"doc_{i}"
    title = metadata.get("title") or metadata.get("doc_id") or ev_id

    return {
        "evidence_id": ev_id,
        "title": title,
        "text": page_content or "",
        "metadata": metadata,
    }


def _normalize_docs(docs: Sequence[Any]) -> List[Dict[str, Any]]:
    return [_doc_to_evidence(d, i) for i, d in enumerate(docs or [])]


def generate_answer_offline_stub(
    question: str,
    docs: Sequence[Any],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Offline-safe generator. Returns JSON with citations and an explicit
    "insufficient_evidence" mode if retrieval is empty.
    """
    evidences = _normalize_docs(docs)[:k]

    if not evidences:
        return {
            "question": question,
            "answer": "Insufficient evidence in the retrieved corpus to answer confidently.",
            "recommendations": [],
            "citations": [],
            "llm": "offline",
            "insufficient_evidence": True,
        }

    citations = []
    for ev in evidences:
        citations.append(
            {
                "evidence_id": ev.get("evidence_id"),
                "title": ev.get("title"),
            }
        )

    # Keep the stub simple + safe. We don't pretend to "know" anything.
    answer = (
        "I retrieved relevant evidence from the corpus. "
        "Use the cited snippets to make a determination; if you need a model-written summary, "
        "run with llm='openai' (optional) or keep offline mode for strict reproducibility."
    )

    return {
        "question": question,
        "answer": answer,
        "recommendations": [],
        "citations": citations,
        "llm": "offline",
        "insufficient_evidence": False,
    }


def generate_answer_openai(
    question: str,
    docs: Sequence[Any],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Optional OpenAI-backed generator. Safe-by-default:
      - no top-level openai import (CI won't need it)
      - requires OPENAI_API_KEY at runtime
      - forces JSON output with citations
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your shell or pass -e OPENAI_API_KEY in Docker."
        )

    try:
        from openai import OpenAI  # imported only when needed
    except ImportError as e:
        raise RuntimeError(
            "OpenAI support not installed. Install requirements_openai.txt (pip install -r requirements_openai.txt)."
        ) from e

    evidences = _normalize_docs(docs)

    # Build evidence text safely (no f-string backslash expressions)
    lines: List[str] = []
    for ev in evidences:
        ev_id = ev.get("evidence_id", "")
        title = ev.get("title", "")
        text = (ev.get("text") or "").strip()
        lines.append(f"[{ev_id}] {title}\n{text}")
    evidence_block = "\n\n---\n\n".join(lines)

    system_msg = (
        "You are a clinical RAG assistant. "
        "You MUST answer using ONLY the provided evidence. "
        "If evidence is insufficient, say so and do not hallucinate. "
        "Return ONLY valid JSON matching the required schema."
    )

    user_msg = (
        "QUESTION:\n"
        f"{question}\n\n"
        "EVIDENCE (cite by evidence_id):\n"
        f"{evidence_block}\n\n"
        "Return JSON with keys:\n"
        "- question (string)\n"
        "- answer (string)\n"
        "- recommendations (list of strings)\n"
        "- citations (list of objects with evidence_id and optional quote)\n"
        "- llm (string)\n"
        "- insufficient_evidence (boolean)\n"
        "Do not include any extra keys."
    )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content
    # content is JSON string; parse defensively
    import json

    try:
        out = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"OpenAI returned non-JSON content: {content[:200]}...") from e

    # ensure minimal required fields exist
    out.setdefault("question", question)
    out.setdefault("llm", "openai")
    out.setdefault("insufficient_evidence", False)
    out.setdefault("citations", [])

    return out


def generate_answer(
    question: str,
    docs: Sequence[Any],
    llm: str = "offline",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Router used by core scripts and prototypes.
    """
    llm = (llm or "offline").lower().strip()
    if llm == "offline":
        return generate_answer_offline_stub(question, docs)
    if llm == "openai":
        return generate_answer_openai(question, docs, **kwargs)
    raise ValueError(f"Unknown llm mode: {llm}")