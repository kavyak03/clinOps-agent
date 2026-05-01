from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.llm.factory import get_llm
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


app = FastAPI(title="ClinOps Agent", version="0.1.0")


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


def _normalize_citations(citations: Any) -> List[Dict[str, Any]]:
    """
    API schema expects citations as List[Dict[str, Any]].

    Real LLMs may return:
      - ["1", "2"]
      - ["[1]", "[2]"]
      - "1"
      - [{"id": "1"}]

    This function normalizes all of those into a safe list of dictionaries.
    """
    if citations is None:
        return []

    if isinstance(citations, dict):
        return [citations]

    if isinstance(citations, str):
        return [{"id": citations}]

    if isinstance(citations, list):
        normalized: List[Dict[str, Any]] = []
        for item in citations:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"id": str(item)})
        return normalized

    return [{"id": str(citations)}]


def _normalize_uncertainties(uncertainties: Any) -> List[str]:
    """
    API schema expects uncertainties as List[str].

    Real LLMs may return:
      - "Some uncertainty text"
      - ["uncertainty 1", "uncertainty 2"]
      - null

    This function normalizes them safely.
    """
    if uncertainties is None:
        return []

    if isinstance(uncertainties, list):
        return [str(item) for item in uncertainties]

    if isinstance(uncertainties, str):
        return [uncertainties]

    return [str(uncertainties)]


def _normalize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Defensive normalization at API boundary.

    This prevents a real LLM provider from crashing the API when it returns
    semantically useful but slightly schema-invalid JSON.
    """
    return {
        "answer": str(result.get("answer", "")),
        "citations": _normalize_citations(result.get("citations", [])),
        "uncertainties": _normalize_uncertainties(result.get("uncertainties", [])),
        "refusal_reason": result.get("refusal_reason"),
    }


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
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

    store = get_vector_store()
    llm = get_llm()
    tracer = TraceLogger()

    evidence = store.search(req.question, k=req.k)
    result = _normalize_llm_result(
        llm.answer_with_citations(question=req.question, evidence=evidence)
    )

    latency_ms = int((time.time() - started) * 1000)

    tracer.log_run(
        run_id=run_id,
        question=req.question,
        model=getattr(llm, "model_name", "unknown"),
        provider=getattr(llm, "provider_name", "unknown"),
        latency_ms=latency_ms,
        meta={"endpoint": "/ask"},
    )
    tracer.log_retrieval(run_id=run_id, evidence=evidence)

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        evidence=[EvidenceChunk(**e) for e in evidence],
        uncertainties=result["uncertainties"],
        refusal_reason=result["refusal_reason"],
        run_id=run_id,
    )


@app.post("/agent/ask", response_model=AskResponse)
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
    llm = get_llm()
    tracer = TraceLogger()

    raw_result = agent_answer(
        question=req.question,
        k=req.k,
        retriever=store,
        llm=llm,
        run_id=run_id,
        tracer=tracer,
    )
    result = _normalize_llm_result(raw_result)

    latency_ms = int((time.time() - started) * 1000)
    tracer.log_run(
        run_id=run_id,
        question=req.question,
        model=getattr(llm, "model_name", "unknown"),
        provider=getattr(llm, "provider_name", "unknown"),
        latency_ms=latency_ms,
        meta={"endpoint": "/agent/ask"},
    )

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        evidence=[EvidenceChunk(**e) for e in raw_result.get("evidence", [])],
        uncertainties=result["uncertainties"],
        refusal_reason=result["refusal_reason"],
        run_id=run_id,
    )


@app.get("/runs/recent")
def runs_recent(limit: int = 20) -> List[Dict[str, Any]]:
    tracer = TraceLogger()
    return tracer.get_recent_runs(limit=limit)


@app.post("/decision/ask", response_model=DecisionResponse)
def decision_ask(req: DecisionRequest) -> DecisionResponse:
    run_id = str(uuid.uuid4())
    started = time.time()

    cohort = req.cohort or generate_synthetic_cohort(40)
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
    llm = get_llm()
    tracer = TraceLogger()
    reranker = get_reranker()

    if req.use_agent:
        rag_result_raw = agent_answer(
            question=question,
            k=req.k,
            retriever=store,
            llm=llm,
            run_id=run_id,
            tracer=tracer,
        )
        rag_result = _normalize_llm_result(rag_result_raw)
        candidate_insight = rag_result["answer"]
        evidence = rag_result_raw.get("evidence", [])
    else:
        candidate_k = max(req.k, getattr(reranker, "candidate_k", req.k))
        evidence = store.search(question, k=candidate_k)
        evidence = reranker.rerank(question, evidence, top_k=req.k)

        rag_result_raw = llm.answer_with_citations(question=question, evidence=evidence)
        rag_result = _normalize_llm_result(rag_result_raw)
        candidate_insight = rag_result["answer"]

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
        meta={"endpoint": "/decision/ask"},
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