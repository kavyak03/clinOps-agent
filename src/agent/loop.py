from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.tools import TOOLS
from src.tracing.logger import TraceLogger
from src.retrieval.reranker import get_reranker


def agent_answer(
    question: str,
    k: int,
    retriever,
    llm,
    run_id: str,
    tracer: Optional[TraceLogger] = None,
) -> Dict[str, Any]:
    tracer = tracer or TraceLogger()

    plan = llm.plan(question)
    query = plan.get("query") or question
    tool_steps = plan.get("tools") or []

    reranker = get_reranker()
    candidate_k = max(k, getattr(reranker, "candidate_k", k))
    evidence = retriever.search(query, k=candidate_k)
    evidence = reranker.rerank(query, evidence, top_k=k)
    tracer.log_retrieval(run_id=run_id, evidence=evidence)

    tool_results: List[Dict[str, Any]] = []
    for step in tool_steps:
        name = (step.get("name") or "").strip()
        params = step.get("input") or {}
        tool_fn = TOOLS.get(name)
        if not tool_fn:
            tool_results.append({"tool": name, "error": "unknown_tool"})
            continue
        out = tool_fn(params, evidence)
        tool_results.append({"tool": name, "output": out})
        tracer.log_tool(run_id=run_id, tool_name=name, inputs=params, outputs=out)

    result = llm.synthesize(question=question, evidence=evidence, tool_results=tool_results)
    result["evidence"] = evidence
    return result
