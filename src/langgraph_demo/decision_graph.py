from __future__ import annotations

from typing import Any, Dict

from src.langgraph_demo.state import DecisionGraphState
from src.simulation.cohort_generator import generate_synthetic_cohort
from src.models.baseline_models import run_baseline_models
from src.validation.statistical_checks import run_statistical_checks
from src.validation.biological_checks import run_biological_checks
from src.validation.cohort_consistency import run_cohort_consistency_checks
from src.validation.evidence_alignment import run_evidence_alignment
from src.validation.decision_gate import make_validation_decision
from src.decision.formatter import format_final_output
from src.decision.recommendation import make_recommendation
from src.decision.explanation import build_explanation


def load_or_generate_cohort(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["cohort"] = state.get("cohort") or generate_synthetic_cohort(40)
    return state


def run_model_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["model_output"] = run_baseline_models(state["cohort"])
    return state


def retrieve_demo_evidence_node(state: DecisionGraphState) -> DecisionGraphState:
    """
    Demo-only evidence retrieval.

    This optional LangGraph example intentionally avoids requiring Postgres,
    pgvector, Docker, or an external LLM. The production/default API path still
    uses the repo's custom retriever + reranker.
    """
    state = dict(state)
    state["evidence"] = [
        {
            "doc_id": "demo-egfr",
            "title": "Demo eGFR evidence",
            "chunk": "Lower eGFR is associated with worse kidney function and higher CKD progression risk.",
            "score": 1.0,
        },
        {
            "doc_id": "demo-a1c",
            "title": "Demo A1c evidence",
            "chunk": "Higher A1c reflects poorer glycemic control and may increase diabetic kidney disease risk.",
            "score": 0.9,
        },
    ]
    return state


def generate_candidate_insight_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["candidate_insight"] = (
        "Lower eGFR and higher A1c are associated with higher CKD-related risk in this cohort."
    )
    return state


def statistical_validation_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["statistical_checks"] = run_statistical_checks(state["model_output"]["scored_cohort"])
    return state


def biological_validation_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["biological_checks"] = run_biological_checks(state["candidate_insight"])
    return state


def cohort_validation_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["cohort_consistency"] = run_cohort_consistency_checks(state["cohort"])
    return state


def evidence_alignment_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["evidence_alignment"] = run_evidence_alignment(state["candidate_insight"], state["evidence"])
    return state


def decision_gate_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    state["validation_summary"] = make_validation_decision(
        state["statistical_checks"],
        state["biological_checks"],
        state["cohort_consistency"],
        state["evidence_alignment"],
    )
    return state


def format_decision_node(state: DecisionGraphState) -> DecisionGraphState:
    state = dict(state)
    final_output = format_final_output(
        state["candidate_insight"],
        state["validation_summary"],
        state["evidence"],
        state["statistical_checks"],
        state["biological_checks"],
        state["cohort_consistency"],
    )
    final_output["final_decision"] = make_recommendation(final_output)
    final_output["explanation"] = build_explanation(state["candidate_insight"], state["validation_summary"])
    state["final_output"] = final_output
    return state


def run_sequential_demo(question: str | None = None) -> Dict[str, Any]:
    state: DecisionGraphState = {"question": question or "Demo LangGraph decision workflow"}
    for node in [
        load_or_generate_cohort,
        run_model_node,
        retrieve_demo_evidence_node,
        generate_candidate_insight_node,
        statistical_validation_node,
        biological_validation_node,
        cohort_validation_node,
        evidence_alignment_node,
        decision_gate_node,
        format_decision_node,
    ]:
        state = node(state)
    state["used_langgraph_runtime"] = False
    return dict(state)


def build_graph():
    """
    Build the optional LangGraph workflow if langgraph is installed.

    Install optional dependencies:
      pip install -r requirements_langchain.txt
    """
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:
        raise RuntimeError(
            "LangGraph is not installed. Install optional deps with "
            "`pip install -r requirements_langchain.txt`, or use run_sequential_demo()."
        ) from exc

    graph = StateGraph(DecisionGraphState)
    graph.add_node("load_or_generate_cohort", load_or_generate_cohort)
    graph.add_node("run_model", run_model_node)
    graph.add_node("retrieve_demo_evidence", retrieve_demo_evidence_node)
    graph.add_node("generate_candidate_insight", generate_candidate_insight_node)
    graph.add_node("statistical_validation", statistical_validation_node)
    graph.add_node("biological_validation", biological_validation_node)
    graph.add_node("cohort_validation", cohort_validation_node)

    # Node name changed to avoid collision with state["evidence_alignment"]
    graph.add_node("compute_evidence_alignment", evidence_alignment_node)

    graph.add_node("decision_gate", decision_gate_node)
    graph.add_node("format_decision", format_decision_node)

    graph.set_entry_point("load_or_generate_cohort")
    graph.add_edge("load_or_generate_cohort", "run_model")
    graph.add_edge("run_model", "retrieve_demo_evidence")
    graph.add_edge("retrieve_demo_evidence", "generate_candidate_insight")
    graph.add_edge("generate_candidate_insight", "statistical_validation")
    graph.add_edge("statistical_validation", "biological_validation")
    graph.add_edge("biological_validation", "cohort_validation")

    # Updated edge references
    graph.add_edge("cohort_validation", "compute_evidence_alignment")
    graph.add_edge("compute_evidence_alignment", "decision_gate")

    graph.add_edge("decision_gate", "format_decision")
    graph.add_edge("format_decision", END)
    return graph.compile()

def run_langgraph_demo(question: str | None = None) -> Dict[str, Any]:
    try:
        app = build_graph()
        out = app.invoke({"question": question or "Demo LangGraph decision workflow"})
        out["used_langgraph_runtime"] = True
        return out
    except RuntimeError:
        # Optional dependency is intentionally not required for the core repo.
        return run_sequential_demo(question=question)
