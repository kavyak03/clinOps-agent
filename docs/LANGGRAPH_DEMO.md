# Optional LangGraph Decision Workflow Demo

The core repo intentionally keeps a transparent custom pipeline as the default:

```text
FastAPI -> retrieval -> reranking -> LLM synthesis -> validation -> decision output
```

This optional demo shows how the same decision workflow maps to LangGraph-style graph orchestration without replacing the main implementation.

## Files

```text
src/langgraph_demo/state.py
src/langgraph_demo/decision_graph.py
scripts/run_langgraph_decision_demo.py
```

## Graph nodes

```text
load_or_generate_cohort
-> run_model
-> retrieve_demo_evidence
-> generate_candidate_insight
-> statistical_validation
-> biological_validation
-> cohort_validation
-> evidence_alignment
-> decision_gate
-> format_decision
```

## Run without installing LangGraph

The script has a fallback sequential runner, so this works even without optional dependencies:

```bash
python scripts/run_langgraph_decision_demo.py
```

It will print:

```json
{"used_langgraph_runtime": false}
```

## Run with LangGraph installed

Install optional framework dependencies:

```bash
pip install -r requirements_langchain.txt
```

Then run:

```bash
python scripts/run_langgraph_decision_demo.py
```

If LangGraph is available, it will print:

```json
{"used_langgraph_runtime": true}
```

## Why this is optional

LangGraph is useful for stateful, branching, multi-step agent workflows. This repo keeps the production/default path custom because the validation and decision logic are easier to inspect and explain directly.
