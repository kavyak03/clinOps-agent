# ClinOps Agent — Biomedical AI Validation & Decision System

An **LLM + RAG + Tools** service for biomedical evidence Q&A.

------------------------------------------------------------------------

## Core Capabilities

-   RAG with citations
-   ReAct-style agent loop (plan → retrieve → tool → synthesize)
-   Safety guardrails for clinical-adjacent use
-   Postgres-backed tracing
-   Evaluation harness for regression testing

------------------------------------------------------------------------

## Quickstart

### Start services

``` bash
docker compose up --build
```

API: http://localhost:8000\
Docs: http://localhost:8000/docs

------------------------------------------------------------------------

### Example query

``` bash
curl -s http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"question":"Summarize evidence about metformin use when eGFR is ~35.", "k": 5}'
```

------------------------------------------------------------------------

## LLM Providers

Default: offline

Anthropic:

``` bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="YOUR_KEY"
```

OpenAI:

``` bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="YOUR_KEY"
```

------------------------------------------------------------------------

## Tracing

Tables: - runs - retrieval_events - tool_events

View recent runs:

``` bash
curl -s "http://localhost:8000/runs/recent?limit=10"
```

------------------------------------------------------------------------

## Evaluation

``` bash
export EVAL_API_BASE=http://localhost:8000
python -m src.eval.run_eval
```

------------------------------------------------------------------------

## AWS Deployment

See deploy/aws/lightsail.md

------------------------------------------------------------------------

## Ingestion (recommended)

Put documents under `data/raw/` (supported: `.txt`, `.md`, `.pdf`) and run:

```bash
# Run ingestion inside the api container (recommended)
docker compose exec api python scripts/ingest.py --input data/raw --glob "**/*.*" --chunk_chars 1200 --overlap 200
```

Then query the API normally (`/ask` or `/agent/ask`). This populates the `embeddings` table in Postgres/pgvector.

## Provider setup (OpenAI / Anthropic) from scratch

This repo runs **offline by default** (no API keys required). To use real LLMs, you only need:
1) an API key from the provider
2) to set environment variables.

### A. Anthropic (recommended)

1) Create an Anthropic account and generate an API key.
2) Set environment variables **in your shell** (works for local runs) **or** in a `.env` file (recommended for Docker).

**Option 1: export in your shell**
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
# Optional model override:
export LLM_MODEL="claude-3-5-sonnet-latest"
```

**Option 2: create a `.env` file (recommended)**
Create a file named `.env` at repo root:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
# Optional:
LLM_MODEL=claude-3-5-sonnet-latest
```
Then run:
```bash
docker compose --env-file .env up --build -d
```

### B. OpenAI

1) Create an OpenAI account and generate an API key.
2) Set environment variables.

**Option 1: export in your shell**
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
# Optional model override:
export LLM_MODEL="gpt-4.1-mini"
```

**Option 2: create a `.env` file (recommended)**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_OPENAI_KEY
# Optional:
LLM_MODEL=gpt-4.1-mini
```
Then:
```bash
docker compose --env-file .env up --build -d
```

### How the provider is used in the pipeline

At request time:
- `/ask` does: **retrieve → synthesize**
- `/agent/ask` does: **plan → retrieve → tool(s) → synthesize**

The LLM provider is selected by:
- `LLM_PROVIDER` = `offline` | `anthropic` | `openai`
- `LLM_MODEL` (optional) to override the default model for that provider

If keys are missing, the API will error at startup for that provider.
If you want a safe default, keep `LLM_PROVIDER=offline` until you’ve set keys.

**Tip:** Start offline, ingest your corpus, then flip providers:
```bash
make up
make ingest
# then set provider envs and restart
docker compose down
docker compose --env-file .env up --build -d
```
------------------------------------------------------------------------

## Cross-encoder re-ranking

This repo now supports a **two-stage retrieval pipeline**:

1. **Broad vector retrieval** with pgvector to maximize recall
2. **Cross-encoder re-ranking** to improve precision before synthesis

Pipeline:

``` text
query
  -> embedding
  -> pgvector search (top-N candidates)
  -> cross-encoder rerank
  -> top-k evidence
  -> LLM synthesis
```

### Why this matters

Vector retrieval is fast, but approximate. A cross-encoder jointly scores the **query and each retrieved chunk together**, which often improves retrieval precision and final answer grounding.

### Environment variables

``` env
RERANK_ENABLE=true
RERANK_CANDIDATES=20
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

If the reranker cannot load for any reason, the system safely falls back to the original vector-ranked order.

---

## Makefile shortcuts

If you have `make` installed, you can run common commands quickly:

```bash
make up        # start services
make ingest    # ingest docs from data/raw into pgvector
make eval      # run eval harness
make logs      # follow api logs
make down      # stop services
```

## Generic CI/CD

This repo includes a generic GitHub Actions CI/CD setup that does not require AWS or other cloud credentials.

Included files:

`.github/workflows/ci.yml` — runs checks and builds the Docker image on push / PR
`.github/workflows/package-image.yml` — packages the Docker image as a downloadable artifact
`.github/workflows/deploy-selfhosted.yml` — optional deployment to a self-hosted Docker machine
`docs/CICD.md` — step-by-step CI/CD runbook


------------------------------------------------------------------------

## Unified flagship architecture

This repo keeps the original RAG backbone and adds three new layers around it:

1. **Simulation layer**
   - synthetic cohort generation
   - noise / bias injection
2. **Validation layer**
   - statistical checks
   - biological plausibility checks
   - cohort consistency checks
   - evidence alignment checks
3. **Decision layer**
   - validated / weakly supported / rejected output
   - recommendation
   - explanation

### New end-to-end flow

Input cohort / patient data  
→ synthetic or processed cohort load  
→ baseline risk modeling  
→ RAG retrieves supporting evidence  
→ LLM generates candidate insight  
→ validation engine checks:
  - statistical validity
  - biological plausibility
  - cohort consistency
  - evidence grounding  
→ decision layer returns:
  - validation status
  - recommendation
  - explanation

### New files added

- `src/simulation/`
- `src/models/`
- `src/validation/`
- `src/decision/`
- `scripts/run_pipeline.py`
- `scripts/run_simulation.py`
- `scripts/run_validation.py`
- `scripts/run_decision_demo.py`
- `schemas/clinical_schema.json`
- `schemas/omics_schema.json`
- `data/synthetic/sample_clinical_cohort.jsonl`

### New API endpoint

- `POST /decision/ask`

This endpoint accepts a cohort + question (or generates a lightweight synthetic cohort if none is provided), runs retrieval and LLM synthesis, validates the candidate insight, and returns a structured decision-grade output.

### New local demo commands

```bash
make run-sim
make run-validate
make run-decision
make run-pipeline
```


------------------------------------------------------------------------

## Production hardening additions

This repo now includes several production-minded safeguards:

- provider error handling for OpenAI/Anthropic failures
- strict LLM output schema normalization via `src/llm/schema.py`
- optional API-key protection with `REQUIRE_API_KEY=true`
- `APP_ENV=production` mode that disables synthetic cohort fallback
- CI smoke eval gate via `python -m src.eval.eval_gate --smoke`
- improved decision evidence formatting with vector/rerank score fields

See:

```text
docs/PRODUCTION_HARDENING.md
docs/LOCAL_NO_DOCKER_TESTING.md
```

### Optional API key auth

```env
REQUIRE_API_KEY=true
API_KEY=your_internal_key
```

Then call protected endpoints with:

```bash
curl -H "X-API-Key: your_internal_key" ...
```

### Production mode

```env
APP_ENV=production
```

In production mode, `/decision/ask` requires explicit `cohort` input and will not silently generate a synthetic demo cohort.

------------------------------------------------------------------------

## Optional LangGraph decision workflow demo

The default pipeline remains custom and transparent. An optional LangGraph-style workflow demo is included for framework familiarity without rewriting the production path.

Run without Docker:

```bash
python scripts/run_langgraph_decision_demo.py
python scripts/run_pipeline.py --offline-demo
```

Install optional dependencies if you want to run with actual LangGraph:

```bash
pip install -r requirements_langchain.txt
python scripts/run_langgraph_decision_demo.py
python scripts/run_pipeline.py --offline-demo
```

See:

```text
docs/LANGGRAPH_DEMO.md
```

------------------------------------------------------------------------

## No-Docker local smoke test

For low-resource machines, run these before pushing:

```bash
python -m compileall src scripts
python scripts/local_smoke_no_docker.py
python -m src.eval.eval_gate --smoke
python scripts/run_langgraph_decision_demo.py
python scripts/run_pipeline.py --offline-demo
```

Full `/ask`, `/agent/ask`, and `/decision/ask` API tests still require pgvector/Postgres, usually via Docker Compose.

