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

This repo supports a **two-stage retrieval pipeline**:

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