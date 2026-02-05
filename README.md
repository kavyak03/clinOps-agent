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

There are **three supported ways** to run this repo. They are **for different usecases** — pick the one that fits your goal.

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

> **Windows PowerShell note**
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
docker run --rm clinrag:latest   python -m scripts.rag_cli   --question "Is metformin appropriate if eGFR is 35?"
```

> **Note:** When using the Docker image **without volume mounts**, the FAISS index is already baked into the image at build time, so no local indexing step is required.

This is the **fastest sanity check** for the core RAG pipeline.

---

## OpenAI integration (optional)

Offline mode is the default. OpenAI is **optional** and never required for CI.

### Step-by-step: set `OPENAI_API_KEY`

#### Option 1 — Set it in your shell (recommended)
**Windows PowerShell**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**Linux / macOS / WSL**
```bash
export OPENAI_API_KEY="sk-..."
```

#### Option 2 — Pass it at Docker run time
**PowerShell**
```powershell
docker run --rm -p 8080:8080 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  clinrag:latest
```

**Linux / macOS / WSL**
```bash
docker run --rm -p 8080:8080   -e OPENAI_API_KEY=$OPENAI_API_KEY   clinrag:latest
```

### API call with `llm=openai`
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
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_langchain.txt   # optional
pip install -r requirements_openai.txt      # optional
```

### Choose data mode

#### Synthetic (offline)
```bash
python -m scripts.make_data
python -m scripts.run_qc
python -m scripts.build_index
```

#### Public PubMedQA
```bash
python -m scripts.download_public_corpus_pubmedqa   --config pqa_labeled --split train --max_examples 2000
python -m scripts.build_index   --corpus data/corpora/public/pubmedqa_corpus_singleline.jsonl
```

### Run RAG CLI locally (after indexing)
```bash
python -m scripts.rag_cli   --question "Is metformin appropriate if eGFR is 35?"
python -m scripts.rag_cli   --question "Is metformin appropriate if eGFR is 35?" --llm openai --model gpt-4o-mini
```

### Run API locally
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8080
```

---

## Evaluation (offline + deterministic)

### Local Python (fast iteration)
```bash
python -m scripts.eval_retrieval
python -m scripts.eval_generation_heuristics
python -m scripts.eval_faithfulness_strict
python -m scripts.make_leaderboard
```

### Run eval inside Docker (recommended for reviewers)
Evaluation scripts run **batch metrics** by executing the retrieval/generation pipeline programmatically. They **do not require** the API server (`curl`) to be running.

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

These are **optional orchestration demos**. The official evaluation harness remains the `scripts/eval_*` scripts so results stay reproducible.

### LangChain (sanity-check wiring)
```bash
docker run --rm clinrag:latest   python -m prototypes.langchain_rag_prototype   --question "Is metformin appropriate if eGFR is 35?"
docker run --rm clinrag:latest   python -m prototypes.langchain_rag_prototype   --question "..." --llm openai --model gpt-4o-mini
```

### LangGraph (sanity-check wiring)
```bash
docker run --rm clinrag:latest   python -m prototypes.langgraph_rag_prototype   --question "Is metformin appropriate if eGFR is 35?"
docker run --rm clinrag:latest   python -m prototypes.langgraph_rag_prototype   --question "..." --llm openai --model gpt-4o-mini
```

---

## CI + Cloud

- GitHub Actions CI runs on every push
- CI should install only `requirements.txt` + `requirements_api.txt`
- Azure deployment is optional and manual
- Docker image is fully self-contained

---

## License
MIT