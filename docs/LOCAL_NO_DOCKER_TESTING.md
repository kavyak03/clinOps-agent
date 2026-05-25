# Local Testing Without Docker

This guide is for low-resource local validation when you do not want to rebuild Docker images.

## What you can test without Docker

You can test:
- Python syntax / compilation
- LLM output schema normalization
- simulation layer
- baseline modeling layer
- validation engine
- decision layer
- CI eval gate smoke mode
- optional LangGraph workflow demo

You cannot fully test the live RAG API without either:
- Docker Compose running Postgres/pgvector, or
- a locally installed PostgreSQL + pgvector database configured through `DATABASE_URL`.

## Setup

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional LangGraph dependencies:

```bash
pip install -r requirements_langchain.txt
```

## No-Docker smoke tests

Run:

```bash
python -m compileall src scripts
python scripts/local_smoke_no_docker.py
python -m src.eval.eval_gate --smoke
python scripts/run_simulation.py
python scripts/run_validation.py
python scripts/run_decision_demo.py
python scripts/run_langgraph_decision_demo.py
python scripts/run_pipeline.py --offline-demo
```

Expected:
- no Python exceptions
- smoke gate passes
- decision demo prints `validation_status`
- LangGraph demo prints final decision output

## API/RAG tests still require a vector database

Endpoints such as:

```text
/ask
/agent/ask
/decision/ask
```

depend on retrieval from pgvector by default. Use Docker or configure `DATABASE_URL` to a local PostgreSQL/pgvector instance before testing those endpoints.
