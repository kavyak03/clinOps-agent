# ClinRAG — Healthcare RAG with Synthetic Data + Public Corpora
**Bioinformatics × LLM Engineering**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Offline](https://img.shields.io/badge/offline-safe-success)

ClinRAG is a **reproducible healthcare Retrieval-Augmented Generation (RAG) mini-system** demonstrating how to build **grounded, auditable, and safe clinical LLM workflows** using **synthetic data** and **public corpora**.

The system is **offline-first by default** and supports optional LLM, orchestration, and cloud integrations.

---

## What this repo demonstrates

- Synthetic patient charts (no PHI)
- Public corpus option (PubMedQA)
- Embeddings + FAISS retrieval
- Structured JSON outputs with citations
- Faithfulness + grounding evaluation harness
- Offline-safe mode (no API calls required)
- Dockerized, reproducible builds
- Optional LangChain + LangGraph prototypes
- Optional Azure cloud deployment

---

## ⭐ Recommended execution paths (important)

There are **three supported ways** to run this repo. They are **for different use cases** — pick the one that fits your goal.

### ✅ Path A — Docker (Recommended)
- Fastest way to verify functionality
- No Python dependency issues
- FAISS index baked into the image
- Ideal for reviewers and demos

### ✅ Path B — Local Python via WSL (Windows) or native Linux/macOS
- Best for development and debugging
- Real `curl`, Linux-like behavior
- Matches cloud + CI environments

### ⚠️ Path C — Local Python via Windows PowerShell
- Supported but more fragile
- PowerShell aliases and quoting differences
- Use only if WSL/Docker are unavailable

---

## Environment variables (what, where, and when)

### 1) Runtime-only variables (most important)
These are **NOT** needed to build the Docker image, and should be set **when you run** the app/CLI.

- `OPENAI_API_KEY` (optional): only needed if you call the API with `"llm":"openai"` or run prototypes with `--llm openai`.

**Windows PowerShell**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**Linux / macOS / WSL**
```bash
export OPENAI_API_KEY="sk-..."
```

**Docker run**
```bash
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY clinrag:latest
```

### 2) Build-time variables (optional)
This repo does not require special build-time env vars. The Dockerfile sets internal defaults like:
- `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, `PIP_NO_CACHE_DIR`

You typically do **not** need to change these.

### 3) Cloud env vars (Azure Container Apps)
If you deploy to Azure Container Apps and want OpenAI-enabled generation:

- Store `OPENAI_API_KEY` as a **Container App secret**
- Map it to an **environment variable**

Example (conceptual):
```bash
az containerapp secret set -n <APP> -g <RG> --secrets OPENAI_API_KEY="sk-..."
az containerapp update -n <APP> -g <RG> --set-env-vars OPENAI_API_KEY=secretref:OPENAI_API_KEY
```

> If you only want offline mode, you do **not** need any OpenAI secrets at all.

---

## 1-minute quickstart (Docker — recommended)

### Build (bakes PubMedQA + FAISS index into the image)
```bash
docker build -t clinrag:latest .
```

### Run API
```bash
docker run --rm -p 8080:8080 clinrag:latest
```

### Test (Linux / macOS / WSL)
```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/ask   -H "Content-Type: application/json"   -d '{"question":"Is metformin appropriate if eGFR is 35?","llm":"offline","k":5}'
```

> **Windows PowerShell note** (use PowerShell-native HTTP)
```powershell
Invoke-RestMethod `
  -Uri http://localhost:8080/ask `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"Is metformin appropriate if eGFR is 35?","llm":"offline","k":5}'
```

---

## 🔹 Run RAG directly from the CLI (no API server)

This runs the **same retrieval + generation logic** as the API, without starting FastAPI.

### Docker (offline)
```bash
docker run --rm clinrag:latest python -m scripts.rag_cli --question "Is metformin appropriate if eGFR is 35?"
```

> **Note:** When using the Docker image **without volume mounts**, the FAISS index is already baked into the image at build time, so **no local indexing step is required**.

This is the **fastest sanity check** for the core RAG pipeline.

---

## Why this project exists

Healthcare LLM systems must be:
- grounded
- auditable
- reproducible
- safe against hallucinations

This repo demonstrates real engineering patterns:
- retrieval over approved corpora only
- answers must cite evidence
- strict structured output
- offline fallback mode
- explicit evaluation harness

---

## Repository structure

```
app/                  FastAPI service
src/                  core RAG + QC logic
scripts/              CLIs (data/QC/index/demo/eval)
prototypes/           LangChain + LangGraph demos
data/
  raw/                synthetic patients/notes
  corpora/            synthetic or public corpora
  processed/          FAISS index + metadata
reports/              evaluation outputs
.github/workflows/     CI workflows
Dockerfile             reproducible container build
docs/                  extra notes
```

---

## OpenAI integration (optional)

Offline mode is the default. To enable OpenAI generation:

1) Set `OPENAI_API_KEY` (see **Environment variables** above)  
2) Run the API container with `-e OPENAI_API_KEY=...`  
3) Call with `llm=openai`:

```bash
curl -X POST http://localhost:8080/ask   -H "Content-Type: application/json"   -d '{"question":"Is metformin appropriate if eGFR is 35?","llm":"openai","k":5}'
```

> OpenAI requires API billing/quota. Offline mode always works.

---

## Optional: Local Python run (no Docker)

### Windows users: use WSL (recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# core deps
pip install -r requirements.txt

# optional layers (only if you need them)
pip install -r requirements_api.txt
pip install -r requirements_langchain.txt
pip install -r requirements_openai.txt
```

### Choose data mode

#### Synthetic (offline)
```bash
python -m scripts.make_data
python -m scripts.run_qc
python -m scripts.build_index
```

#### Public PubMedQA (public + reproducible)
```bash
python -m scripts.download_public_corpus_pubmedqa --config pqa_labeled --split train --max_examples 2000
python -m scripts.build_index --corpus data/corpora/public/pubmedqa_corpus_singleline.jsonl
```

### Run RAG CLI locally (after indexing)
```bash
python -m scripts.rag_cli --question "Is metformin appropriate if eGFR is 35?"
```

### Run API locally
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8080
```

---

## Evaluation (offline + deterministic)

Evaluation scripts run **batch metrics** by executing the retrieval/generation pipeline programmatically. They **do not require** the API server (`curl`) to be running.

### Local Python (fast iteration)
```bash
python -m scripts.eval_retrieval
python -m scripts.eval_generation_heuristics
python -m scripts.eval_faithfulness_strict
python -m scripts.make_leaderboard
```

### Run eval inside Docker (recommended for reviewers)
```bash
docker run --rm clinrag:latest python -m scripts.eval_retrieval
docker run --rm clinrag:latest python -m scripts.eval_generation_heuristics
docker run --rm clinrag:latest python -m scripts.eval_faithfulness_strict
docker run --rm clinrag:latest python -m scripts.make_leaderboard
```

### Save evaluation outputs to your machine (mount `reports/`)

**WSL / Linux / macOS**
```bash
mkdir -p reports

docker run --rm -v "$(pwd)/reports:/app/reports" clinrag:latest python -m scripts.eval_retrieval
docker run --rm -v "$(pwd)/reports:/app/reports" clinrag:latest python -m scripts.eval_generation_heuristics
docker run --rm -v "$(pwd)/reports:/app/reports" clinrag:latest python -m scripts.eval_faithfulness_strict
docker run --rm -v "$(pwd)/reports:/app/reports" clinrag:latest python -m scripts.make_leaderboard
```

**Windows PowerShell**
```powershell
New-Item -ItemType Directory -Force reports | Out-Null

docker run --rm `
  -v ${PWD}
eports:/app/reports `
  clinrag:latest python -m scripts.eval_retrieval
```
(Repeat the same pattern for the other eval scripts.)

---

## LangChain + LangGraph prototypes (optional)

These are **optional orchestration demos**. The **official evaluation harness** remains the `scripts/eval_*` scripts so results stay reproducible and deterministic.

Install optional deps:
```bash
pip install -r requirements_langchain.txt
```

### Sanity-check LangChain wiring (not metrics)
```bash
docker run --rm clinrag:latest python -m prototypes.langchain_rag_prototype --question "Is metformin appropriate if eGFR is 35?"
```

### Sanity-check LangGraph wiring (not metrics)
```bash
docker run --rm clinrag:latest python -m prototypes.langgraph_rag_prototype --question "Is metformin appropriate if eGFR is 35?"
```

> Note: The eval scripts evaluate the core retrieval/generation pipeline directly (faster, deterministic). The prototypes are for demonstrating framework familiarity.

---

## CI + Cloud

- GitHub Actions CI runs on every push and does a Python import/compile sanity check + Docker build
- OpenAI is **not required** for CI (no secrets needed unless you add OpenAI tests)
- Azure deployment is optional and manual
- Docker image is fully self-contained (with baked retrieval index)

---

## License
MIT