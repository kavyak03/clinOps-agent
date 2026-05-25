from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class LLMAnswer(BaseModel):
    """Canonical provider output used at the API boundary."""

    answer: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    provider_error: Optional[Dict[str, Any]] = None


def normalize_citations(citations: Any) -> List[Dict[str, Any]]:
    if citations is None:
        return []
    if isinstance(citations, dict):
        return [citations]
    if isinstance(citations, str):
        return [{"id": citations}]
    if isinstance(citations, list):
        out: List[Dict[str, Any]] = []
        for item in citations:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"id": str(item)})
        return out
    return [{"id": str(citations)}]


def normalize_uncertainties(uncertainties: Any) -> List[str]:
    if uncertainties is None:
        return []
    if isinstance(uncertainties, list):
        return [str(item) for item in uncertainties]
    if isinstance(uncertainties, str):
        return [uncertainties]
    return [str(uncertainties)]


def normalize_llm_answer(raw: Any) -> LLMAnswer:
    """Convert unpredictable provider/model output into the canonical schema."""
    if isinstance(raw, LLMAnswer):
        return raw

    if not isinstance(raw, dict):
        return LLMAnswer(
            answer=str(raw),
            citations=[],
            uncertainties=["LLM output was not a dictionary-like object."],
        )

    candidate = {
        "answer": str(raw.get("answer", "")),
        "citations": normalize_citations(raw.get("citations", [])),
        "uncertainties": normalize_uncertainties(raw.get("uncertainties", [])),
        "refusal_reason": raw.get("refusal_reason"),
        "provider_error": raw.get("provider_error"),
    }

    try:
        return LLMAnswer(**candidate)
    except ValidationError as exc:
        return LLMAnswer(
            answer=str(raw.get("answer", "")),
            citations=[],
            uncertainties=[f"LLM output failed schema validation: {exc}"],
            provider_error={"type": "schema_validation_error"},
        )


def provider_error_answer(provider: str, model: str, exc: Exception) -> Dict[str, Any]:
    """Return a safe, schema-compatible response when an external provider fails."""
    return LLMAnswer(
        answer=(
            f"{provider} provider error. Retrieval and validation components may have run, "
            "but external LLM synthesis could not complete."
        ),
        citations=[],
        uncertainties=[
            "External LLM provider call failed. Check API key, quota, billing, model access, or network connectivity."
        ],
        provider_error={
            "provider": provider,
            "model": model,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        },
    ).model_dump()
