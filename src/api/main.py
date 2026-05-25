from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.llm.factory import get_llm
from src.llm.schema import normalize_llm_answer
from src.vectorstore.factory import get_vector_store
from src.agent.loop import agent_answer
from src.tracing.logger import TraceLogger
from src.retrieval.reranker import get_reranker

from src.validation.statistical_checks import run_statistical_checks
from src.validation.cohort_consistency import run_cohort_consistency_checks
from src.validation.biological_checks import run_biological_checks
from src.validation.evidence_alignment import run_evidence_alignment
from src.validation.decision_gate import make_validation_decision
from src.decision.formatter import format_final_output
from src.decision.recommendation import make_recommendation
from src.decision.explanation import build_explanation
from src.models.baseline_models import run_baseline_models
from src.simulation.cohort_generator import generate_synthetic_cohort


app = FastAPI(title="ClinOps Agent", version="0.2.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class EvidenceChunk(BaseModel):
    doc_id: str
    title: Optional[str] = None
    chunk: str
    score: Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[EvidenceChunk] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    run_id: str


class DecisionRequest(BaseModel):
    cohort: Optional[List[Dict[str, Any]]] = None
    question: Optional[str] = None
    use_agent: bool = False
    k: int = Field(default=5, ge=1, le=20)


class DecisionResponse(BaseModel):
    candidate_insight: str
    validation_status: str
    confidence: str
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    statistical_checks: Dict[str, Any]
    biological_checks: Dict[str, Any]
    cohort_consistency: Dict[str, Any]
    final_decision: str
    explanation: str
    run_id: str
    clinical_use_boundary: str = (
        "Research and decision-support only. Not medical advice. "
        "Final clinical decisions require qualified professional review."
    )


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _app_env() -> str:
    return os.getenv("APP_ENV", "local").strip().lower()


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Optional lightweight API-key protection.

    Local/default behavior:
      REQUIRE_API_KEY=false -> no auth required

    Production-ish behavior:
      REQUIRE_API_KEY=true and API_KEY=<secret>
      Clients must send: X-API-Key: <secret>
    """
    if not _truthy_env("REQUIRE_API_KEY", "false"):
        return

    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="API key auth is enabled but API_KEY is not configured.")

    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _basic_guardrails(question: str) -> Optional[str]:
    q = question.lower()
    red_flags = [
        "dose",
        "dosage",
        "how much should i take",
        "prescribe",
        "should i take",
        "can i take",
        "diagnose",
        "diagnosis",
        "emergency",
        "urgent",
        "chest pain",
        "shortness of breath",
    ]
    if any(rf in q for rf in red_flags):
        return (
            "I can’t provide medical diagnosis, dosing, or urgent medical advice. "
            "I can summarize evidence and suggest discussing with a licensed clinician."
        )
    return None


def _get_llm_or_http():
    try:
        return get_llm()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_type": "llm_provider_configuration_error",
                "message": str(exc),
                "hint": "Check LLM_PROVIDER, provider API keys, LLM_MODEL, quota, billing, and model access.",
            },
        ) from exc


def _retrieve_with_rerank(question: str, k: int) -> List[Dict[str, Any]]:
    store = get_vector_store()
    reranker = get_reranker()
    candidate_k = max(k, getattr(reranker, "candidate_k", k))
    evidence = store.search(question, k=candidate_k)
    evidence = reranker.rerank(question, evidence, top_k=k)
    return evidence


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "app_env": _app_env()}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest) -> AskResponse:
    run_id = str(uuid.uuid4())
    started = time.time()

    refusal = _basic_guardrails(req.question)
    if refusal:
        return AskResponse(
            answer=refusal,
            citations=[],
            evidence=[],
            uncertainties=["Safety refusal triggered by guardrails."],
            refusal_reason="medical_advice",
            run_id=run_id,
        )

    llm = _get_llm_or_http()
    tracer = TraceLogger()

    evidence = _retrieve_with_rerank(req.question, req.k)
    result = normalize_llm_answer(
        llm.answer_with_citations(question=req.question, evidence=evidence)
    )

    latency_ms = int((time.time() - started) * 1000)

    tracer.log_run(
        run_id=run_id,
        question=req.question,
        model=getattr(llm, "model_name", "unknown"),
        provider=getattr(llm, "provider_name", "unknown"),
        latency_ms=latency_ms,
        meta={
            "endpoint": "/ask",
            "app_env": _app_env(),
            "provider_error": result.provider_error,
            "evidence_count": len(evidence),
        },
    )
    tracer.log_retrieval(run_id=run_id, evidence=evidence)

    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        evidence=[EvidenceChunk(**e) for e in evidence],
        uncertainties=result.uncertainties,
        refusal_reason=result.refusal_reason,
        run_id=run_id,
    )


@app.post("/agent/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def agent_ask(req: AskRequest) -> AskResponse:
    run_id = str(uuid.uuid4())
    started = time.time()

    refusal = _basic_guardrails(req.question)
    if refusal:
        return AskResponse(
            answer=refusal,
            citations=[],
            evidence=[],
            uncertainties=["Safety refusal triggered by guardrails."],
            refusal_reason="medical_advice",
            run_id=run_id,
        )

    store = get_vector_store()
    llm = _get_llm_or_http()
    tracer = TraceLogger()

    raw_result = agent_answer(
        question=req.question,
        k=req.k,
        retriever=store,
        llm=llm,
        run_id=run_id,
        tracer=tracer,
    )
    result = normalize_llm_answer(raw_result)

    latency_ms = int((time.time() - started) * 1000)
    tracer.log_run(
        run_id=run_id,
        question=req.question,
        model=getattr(llm, "model_name", "unknown"),
        provider=getattr(llm, "provider_name", "unknown"),
        latency_ms=latency_ms,
        meta={
            "endpoint": "/agent/ask",
            "app_env": _app_env(),
            "provider_error": result.provider_error,
            "evidence_count": len(raw_result.get("evidence", [])),
        },
    )

    return AskResponse(
        answer=result.answer,
        citations=result.citations,
        evidence=[EvidenceChunk(**e) for e in raw_result.get("evidence", [])],
        uncertainties=result.uncertainties,
        refusal_reason=result.refusal_reason,
        run_id=run_id,
    )


@app.get("/runs/recent", dependencies=[Depends(require_api_key)])
def runs_recent(limit: int = 20) -> List[Dict[str, Any]]:
    tracer = TraceLogger()
    return tracer.get_recent_runs(limit=limit)


@app.post("/decision/ask", response_model=DecisionResponse, dependencies=[Depends(require_api_key)])
def decision_ask(req: DecisionRequest) -> DecisionResponse:
    run_id = str(uuid.uuid4())
    started = time.time()

    if req.cohort is None:
        if _app_env() == "production":
            raise HTTPException(
                status_code=400,
                detail={
                    "error_type": "cohort_required",
                    "message": "APP_ENV=production requires explicit cohort input. Synthetic fallback is disabled.",
                },
            )
        cohort = generate_synthetic_cohort(40)
    else:
        cohort = req.cohort

    model_out = run_baseline_models(cohort)
    question = req.question or (
        f"Summarize the relationship between eGFR, A1c, and CKD-related risk "
        f"for a cohort with mean predicted risk {model_out['summary'].get('mean_predicted_risk')}."
    )

    refusal = _basic_guardrails(question)
    if refusal:
        return DecisionResponse(
            candidate_insight=refusal,
            validation_status="rejected",
            confidence="low",
            supporting_evidence=[],
            statistical_checks={},
            biological_checks={},
            cohort_consistency={},
            final_decision="Do not use this output for decision support.",
            explanation="The request was blocked by clinical-adjacent safety guardrails.",
            run_id=run_id,
        )

    store = get_vector_store()
    llm = _get_llm_or_http()
    tracer = TraceLogger()

    if req.use_agent:
        rag_result_raw = agent_answer(
            question=question,
            k=req.k,
            retriever=store,
            llm=llm,
            run_id=run_id,
            tracer=tracer,
        )
        rag_result = normalize_llm_answer(rag_result_raw)
        candidate_insight = rag_result.answer
        evidence = rag_result_raw.get("evidence", [])
    else:
        evidence = _retrieve_with_rerank(question, req.k)
        rag_result_raw = llm.answer_with_citations(question=question, evidence=evidence)
        rag_result = normalize_llm_answer(rag_result_raw)
        candidate_insight = rag_result.answer

    statistical = run_statistical_checks(model_out["scored_cohort"])
    biological = run_biological_checks(candidate_insight)
    cohort_checks = run_cohort_consistency_checks(cohort)
    evidence_checks = run_evidence_alignment(candidate_insight, evidence)
    validation_summary = make_validation_decision(
        statistical,
        biological,
        cohort_checks,
        evidence_checks,
    )

    final_output = format_final_output(
        candidate_insight,
        validation_summary,
        evidence,
        statistical,
        biological,
        cohort_checks,
    )
    final_decision = make_recommendation(final_output)
    explanation = build_explanation(candidate_insight, validation_summary)

    latency_ms = int((time.time() - started) * 1000)
    tracer.log_run(
        run_id=run_id,
        question=question,
        model=getattr(llm, "model_name", "unknown"),
        provider=getattr(llm, "provider_name", "unknown"),
        latency_ms=latency_ms,
        meta={
            "endpoint": "/decision/ask",
            "app_env": _app_env(),
            "provider_error": rag_result.provider_error,
            "validation_status": validation_summary["validation_status"],
            "evidence_count": len(evidence),
        },
    )
    tracer.log_retrieval(run_id=run_id, evidence=evidence)

    return DecisionResponse(
        candidate_insight=candidate_insight,
        validation_status=validation_summary["validation_status"],
        confidence=validation_summary["confidence"],
        supporting_evidence=final_output["supporting_evidence"],
        statistical_checks=statistical,
        biological_checks=biological,
        cohort_consistency=cohort_checks,
        final_decision=final_decision,
        explanation=explanation,
        run_id=run_id,
    )
