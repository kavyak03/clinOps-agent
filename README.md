# ClinOps Agent — Biomedical AI Validation & Decision System

A **biomedical LLM + RAG + validation + decision-support system**.

This project goes beyond a standard RAG chatbot. It combines evidence-grounded biomedical retrieval, LLM synthesis with citations, optional tool-using agent workflow, cross-encoder reranking, structured validation gates, decision-grade output, tracing, evaluation, and CI/CD scaffolding.

The core idea:

> Generate biomedical insights, validate them statistically and biologically, and return a transparent decision-support output.

---

## Clinical Use Boundary

This project is for **research and decision-support system design**.

It is **not** a medical device and does **not** provide medical advice, diagnosis, dosing guidance, or treatment recommendations. Final clinical decisions require qualified professional review.

---

## Core Capabilities

| Capability | Description |
|---|---|
| RAG with citations | Retrieves evidence from local biomedical documents and generates citation-grounded answers |
| Agent workflow | Optional ReAct-style loop: plan → retrieve → tool(s) → synthesize |
| Cross-encoder reranking | Retrieves broad candidates with pgvector, then reranks for higher precision |
| Validation engine | Checks statistical validity, biological plausibility, cohort consistency, and evidence alignment |
| Decision layer | Returns `validated`, `weakly_supported`, or `rejected` decision-support output |
| Tracing | Logs runs, retrieval events, and tool events |
| Evaluation | Supports regression/eval checks for retrieval and application behavior |
| Production hardening | Provider error handling, schema normalization, API-key auth option, production mode controls |
| Optional LangGraph demo | Shows how the decision workflow maps to graph-based orchestration |

---

## System Architecture

```text
Structured cohort / patient data
        ↓
Simulation or input cohort loading
        ↓
Baseline risk modeling
        ↓
RAG retrieval from biomedical corpus
        ↓
Cross-encoder reranking
        ↓
LLM candidate insight generation
        ↓
Validation engine
  ├── statistical checks
  ├── biological plausibility checks
  ├── cohort consistency checks
  └── evidence alignment checks
        ↓
Decision layer
        ↓
Validated / weakly supported / rejected output
```

---

## Main API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Basic API health check |
| `POST /ask` | Standard RAG: retrieve → synthesize |
| `POST /agent/ask` | Agentic RAG: plan → retrieve → tools → synthesize |
| `POST /decision/ask` | Full validation workflow: RAG insight → validation gates → decision output |
| `GET /runs/recent` | Inspect recent traced runs |

---

## Local Quickstart with Docker

### 1. Start services

Recommended two-step flow:

```bash
docker compose build api
docker compose up -d
```

Some Docker versions also support:

```bash
docker compose up --build -d
```

API:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl -sS http://localhost:8000/healthz && echo
```

Expected:

```json
{"status":"ok","app_env":"local"}
```

---

## Ingestion

Put small demo documents under:

```text
data/raw/
```

Supported file types:

- `.txt`
- `.md`
- `.pdf`

Run ingestion inside the API container:

```bash
docker compose exec api python scripts/ingest.py \
  --input data/raw \
  --glob "**/*.*" \
  --chunk_chars 1200 \
  --overlap 200
```

This populates the `embeddings` table in Postgres/pgvector.

---

## Example API Calls

### `/ask`

```bash
curl -sS http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does low eGFR relate to CKD risk?", "k": 3}' | python -m json.tool
```

### `/agent/ask`

```bash
curl -sS http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain CKD risk and uncertainty using the retrieved evidence.", "k": 5}' | python -m json.tool
```

### `/decision/ask`

```bash
curl -sS http://localhost:8000/decision/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the relationship between eGFR, A1c, and CKD-related risk in this cohort.", "use_agent": false, "k": 5}' | python -m json.tool
```

Expected decision output includes:

```json
{
  "candidate_insight": "...",
  "validation_status": "validated",
  "confidence": "moderate",
  "supporting_evidence": [],
  "statistical_checks": {},
  "biological_checks": {},
  "cohort_consistency": {},
  "final_decision": "...",
  "explanation": "...",
  "clinical_use_boundary": "Research and decision-support only. Not medical advice. Final clinical decisions require qualified professional review."
}
```

---

## LLM Providers

The repo runs in **offline mode by default**, so no API key is required for local smoke testing.

Provider selection:

```env
LLM_PROVIDER=offline
```

Supported values:

```text
offline | openai | anthropic
```

### OpenAI setup

Create `.env` at repo root:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_OPENAI_KEY
LLM_MODEL=gpt-4o-mini

RERANK_ENABLE=true
RERANK_CANDIDATES=20
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Restart with:

```bash
docker compose down
docker compose --env-file .env up -d --force-recreate
```

Verify inside the container:

```bash
docker compose exec api printenv LLM_PROVIDER
docker compose exec api python -c "import os; print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))"
```

### Anthropic setup

Create `.env`:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
LLM_MODEL=claude-3-5-sonnet-latest

RERANK_ENABLE=true
RERANK_CANDIDATES=20
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Restart:

```bash
docker compose down
docker compose --env-file .env up -d --force-recreate
```

---

## Recommended Provider Workflow

Start offline first:

```bash
docker compose build api
docker compose up -d
docker compose exec api python scripts/ingest.py --input data/raw --glob "**/*.*" --chunk_chars 1200 --overlap 200
```

Then switch to a real provider only after ingestion and retrieval work.

This avoids debugging provider quota, billing, or API-key issues before the local pipeline is validated.

---

## Cross-Encoder Reranking

This repo supports a two-stage retrieval pipeline:

```text
query
  → embedding
  → pgvector search over broad candidates
  → cross-encoder reranking
  → top-k evidence
  → LLM synthesis
```

Environment variables:

```env
RERANK_ENABLE=true
RERANK_CANDIDATES=20
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Score semantics

After reranking:

| Field | Meaning |
|---|---|
| `vector_score` | Original vector retrieval score from pgvector |
| `rerank_score` | Cross-encoder relevance score |
| `score` | Final ranking score; currently set to `rerank_score` after reranking |

Cross-encoder scores can be negative. They are relevance scores, not probabilities. Higher is better.

---

## Validation Engine

The validation layer checks whether an AI-generated biomedical insight is acceptable for decision-support use.

Modules:

```text
src/validation/
  statistical_checks.py
  biological_checks.py
  cohort_consistency.py
  evidence_alignment.py
  decision_gate.py
```

Validation categories:

| Check | Purpose |
|---|---|
| Statistical checks | Sample size, calibration gap, direction sanity |
| Biological checks | Biomarker directionality and plausibility |
| Cohort consistency | Missingness, subgroup balance, sample size |
| Evidence alignment | Whether the generated insight is grounded in retrieved evidence |
| Decision gate | Converts checks into `validated`, `weakly_supported`, or `rejected` |

---

## Decision Layer

Modules:

```text
src/decision/
  formatter.py
  recommendation.py
  explanation.py
```

The decision layer formats the final structured response:

```json
{
  "candidate_insight": "...",
  "validation_status": "validated",
  "confidence": "moderate",
  "supporting_evidence": [],
  "statistical_checks": {},
  "biological_checks": {},
  "cohort_consistency": {},
  "final_decision": "...",
  "explanation": "..."
}
```

---

## Simulation and Baseline Modeling

Simulation modules:

```text
src/simulation/
  cohort_generator.py
  noise_injection.py
  bias_injection.py
  intervention_scenarios.py
```

Model modules:

```text
src/models/
  risk_model.py
  baseline_models.py
  calibration.py
```

These are intentionally lightweight. They exist to support the validation workflow, not to claim clinical-grade predictive modeling.

---

## Optional LangGraph Demo

This repo keeps the custom Python pipeline as the default path for transparency and debuggability.

An optional LangGraph-style decision workflow demo is included to show how the validation workflow maps to graph orchestration concepts.

Run:

```bash
python scripts/run_langgraph_decision_demo.py
```

Relevant files:

```text
src/langgraph_demo/
  state.py
  decision_graph.py

docs/LANGGRAPH_DEMO.md
```

The LangGraph demo is optional and does not replace the default `/decision/ask` implementation.

---

## No-Docker Local Testing

For lightweight checks without Docker:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m compileall src scripts
python scripts/local_smoke_no_docker.py
python -m src.eval.eval_gate --smoke
python scripts/run_simulation.py
python scripts/run_validation.py
python scripts/run_decision_demo.py
python scripts/run_langgraph_decision_demo.py
python scripts/run_pipeline.py --offline-demo
```

These tests validate local Python modules, validation logic, decision formatting, eval gate smoke behavior, and offline pipeline behavior.

They do **not** validate pgvector, Docker networking, or API endpoints.

---

## Docker Smoke Test Before Push

A minimal Docker smoke test:

```bash
docker compose build api
docker compose up -d

curl -sS http://localhost:8000/healthz && echo

docker compose exec api python scripts/ingest.py \
  --input data/raw \
  --glob "**/*.txt" \
  --chunk_chars 1200 \
  --overlap 200

curl -sS http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does low eGFR relate to CKD risk?", "k": 3}' | python -m json.tool

curl -sS http://localhost:8000/decision/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the relationship between eGFR, A1c, and CKD-related risk in this cohort.", "use_agent": false, "k": 5}' | python -m json.tool
```

---

## Production Hardening Features

This repo includes several production-minded controls.

### Provider error handling

Provider failures such as quota, billing, rate limits, malformed JSON, or model errors are handled gracefully instead of crashing the API.

### LLM output schema normalization

LLM outputs are normalized so the API does not crash when a provider returns:

```json
"citations": ["1"]
```

instead of:

```json
"citations": [{"id":"1"}]
```

or:

```json
"uncertainties": "single uncertainty string"
```

instead of:

```json
"uncertainties": ["single uncertainty string"]
```

### API key protection

Optional internal API-key auth can be enabled:

```env
REQUIRE_API_KEY=true
API_KEY=replace_with_internal_key
```

Then include:

```bash
-H "X-API-Key: replace_with_internal_key"
```

### Production mode

```env
APP_ENV=production
```

In production mode, `/decision/ask` should not silently fall back to a synthetic cohort. A real cohort should be supplied explicitly.

### Clinical boundary

Decision responses include a clinical-use boundary indicating that the output is research/decision-support only and not medical advice.

---

## Evaluation

Run API-based eval:

```bash
export EVAL_API_BASE=http://localhost:8000
python -m src.eval.run_eval
```

Run smoke eval gate:

```bash
python -m src.eval.eval_gate --smoke
```

Thresholds live in:

```text
eval/quality_thresholds.json
```

---

## Tracing

Tracing tables include:

```text
runs
retrieval_events
tool_events
```

View recent runs:

```bash
curl -sS "http://localhost:8000/runs/recent?limit=10" | python -m json.tool
```

Tracing helps inspect endpoint used, model/provider, latency, retrieved evidence, tool execution, and validation status.

---

## Makefile Shortcuts

If `make` is installed:

```bash
make up          # start services
make ingest      # ingest docs from data/raw into pgvector
make eval        # run eval harness
make logs        # follow api logs
make down        # stop services
```

Additional useful scripts:

```bash
python scripts/run_simulation.py
python scripts/run_validation.py
python scripts/run_decision_demo.py
python scripts/run_pipeline.py --offline-demo
python scripts/run_langgraph_decision_demo.py
```

---

## Generic CI/CD

This repo includes generic GitHub Actions workflows that do not require AWS or cloud credentials.

Included files:

```text
.github/workflows/ci.yml
.github/workflows/package-image.yml
.github/workflows/deploy-selfhosted.yml
docs/CICD.md
```

Capabilities:

- CI checks on push / PR
- Docker image build validation
- downloadable Docker image artifact packaging
- optional self-hosted deployment workflow

The self-hosted deployment workflow is infrastructure-ready but requires a machine with Docker and a GitHub self-hosted runner.

---

## Security Notes

Do not commit secrets.

Ensure `.gitignore` contains:

```gitignore
.env
.env.*
```

Do not log real PHI or patient identifiers in production. The demo data and example documents are synthetic/local testing artifacts.

---

## Troubleshooting

### `docker compose up --build` fails with `unknown flag: --build`

Use the two-step equivalent:

```bash
docker compose build api
docker compose up -d
```

### API returns offline stub even after setting provider

Verify environment variables inside the container:

```bash
docker compose exec api printenv LLM_PROVIDER
docker compose exec api python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```

Make sure `docker-compose.yml` uses interpolation:

```yaml
LLM_PROVIDER: ${LLM_PROVIDER:-offline}
OPENAI_API_KEY: ${OPENAI_API_KEY:-}
```

### OpenAI returns quota error

This means the key is valid enough to reach OpenAI, but your API account/project does not have usable quota or billing capacity. Switch to offline mode for local testing or fix provider billing.

### Health check returns nothing

Use:

```bash
curl -i http://localhost:8000/healthz
```

Then inspect:

```bash
docker compose ps
docker compose logs --tail=100 api
```

### Reranking appears disabled

Check:

```bash
docker compose exec api printenv RERANK_ENABLE
docker compose exec api printenv RERANK_CANDIDATES
```

A working reranked response should include:

```json
"vector_score": 0.71,
"rerank_score": 6.69,
"score": 6.69
```

---

## Recommended First Demo

1. Add 3–5 small demo `.txt` files to `data/raw/`
2. Start Docker
3. Ingest the docs
4. Run `/ask`
5. Run `/agent/ask`
6. Run `/decision/ask`
7. Inspect `/runs/recent`
8. Run smoke eval

This gives a complete demonstration of:

```text
documents → embeddings → retrieval → reranking → LLM synthesis → validation → decision output
```