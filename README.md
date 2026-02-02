# ClinRAG: Healthcare RAG on Synthetic Patient Charts (Bioinformatics + LLM Engineering)

ClinRAG is a mini-project demonstrating how to build a **healthcare-safe** Retrieval-Augmented Generation (RAG) system using **synthetic clinical data** and a **guideline corpus**. It emphasizes:

- Data QC & cleaning (missingness, ranges, outliers)
- Retrieval with embeddings + FAISS
- Structured JSON outputs with citations
- Basic evaluation for retrieval quality and faithfulness

## Why this project
Healthcare LLM systems must be **grounded, auditable, and safe**. This repo demonstrates engineering patterns used in real systems:
- Retrieval over approved corpora (guidelines, reference notes)
- Strict structured outputs
- Guardrails for insufficient evidence
- Offline-safe mode (no API calls required)

## Repository structure
- `data/raw/` – generated synthetic patients, labs, and notes (JSONL)
- `data/corpora/` – synthetic guideline snippets (JSONL)
- `data/processed/` – chunk metadata + FAISS index
- `reports/` – QC plots + QC summary (created after running QC)
- `src/` – data generation, QC, RAG components
- `scripts/` – CLI scripts for generation, QC, indexing, demo retrieval

## Quickstart

### 1) Install



```bash
#Create a virtual env on Windows powershell. Python 3.11 recommended. On windows machines, Windows: use WSL2 or conda
py -3.11 -m venv .venv-clinrag
.venv-clinrag\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
$env:PYTHONNOUSERSITE="1"
python -m pip cache purge
#install core dependencies as wheels only
python -m pip install --only-binary=:all: --no-cache-dir --force-reinstall `
  "numpy==1.26.4" "pandas==2.2.2" "scikit-learn==1.5.1" "matplotlib==3.9.0" "tqdm" "requests"
#verify imports
python -c "import numpy as np, pandas as pd; print(np.__version__, pd.__version__)"

#Install rest of the requirements
python -m pip install --only-binary=:all: --no-cache-dir -r requirements.txt
python -m pip install -r requirements_openai.txt
```

### 2) Generate synthetic data
```bash
python scripts/make_data.py

#OR on windows powershell outside of the scripts directory, run as -
python -m scripts.make_data
```

### 3) Run QC + plots
```bash
python scripts/run_qc.py
```
Outputs are saved in `reports/` (PNG plots + `qc_summary.json`).

### 4) Build retrieval index
```bash
python scripts/build_index.py
```

### 5) Run a retrieval-only demo (no LLM API required)
```bash
python scripts/rag_cli.py --question "Is metformin appropriate if eGFR is 35?"
```

## Notes on LLM integration
The repo includes `src/rag/generate.py` with a stubbed generator function for offline stubs (generate_answer_offline_stub) and a generator function (generate_answer_openai) where you can plug in an OpenAI LLM model for answer generation to prompts. The prompts enforce:
- Answer using **ONLY** retrieved evidence
- Return **ONLY** valid JSON with citations

### Ensure to set openAI key before using the openAI gnerator function for answers
```bash
$env:OPENAI_API_KEY="sk-..."
```

## Safety
All data is **synthetic** (no PHI).

## License
MIT (adjust as you like).
---

## Run locally (VS Code friendly)
This project runs on a normal laptop with **free software**.

Open the folder in **VS Code**, select the `.venv` interpreter, and run scripts from the terminal.

---

## Data options (no PHI)
You can run this repo with either:

### Option A — Simulated realistic data (default)
- Synthetic patient charts/labs/notes can be generated locally:
```bash
python scripts/make_data.py
python scripts/run_qc.py
```
- Default retrieval corpora are in `data/corpora/` (synthetic guideline-style documents)

### Option B — Public corpus mode (optional)
Download a **public** PubMed QA corpus (PubMedQA) and use it as the retrieval corpus:
```bash
python scripts/download_public_corpus_pubmedqa.py
python scripts/build_index.py --corpus data/corpora/public/pubmedqa_corpus.jsonl
```

> Public corpus download requires internet access at runtime. Everything else stays free.

---

## Build the index
Default corpus:
```bash
python scripts/build_index.py
```

Alternative corpus:
```bash
python scripts/build_index.py --corpus data/corpora/public/pubmedqa_corpus.jsonl
```

---

## Evaluation

### 1) Retrieval evaluation (Recall@k, MRR@k)
```bash
python scripts/eval_retrieval.py
```

**Recall@k**: did we retrieve *any* gold document in top-k?  
**MRR@k**: how early did the first gold doc appear (1.0 best; 0.5 if rank 2, etc.)

Outputs:
- `reports/retrieval_eval_summary.json`
- `reports/retrieval_eval_per_question.json`

Gold set:
- `data/corpora/questions_gold.jsonl`

### 2) Generation heuristics (fast, reproducible)
```bash
python scripts/eval_generation_heuristics.py
```

- **citation_coverage**: fraction of recommendations that have at least one citation
- **faithfulness_overlap**: rough keyword overlap between answer and evidence

Output:
- `reports/generation_heuristics.json`

### 3) Stricter faithfulness checks (citation grounding)
```bash
python scripts/eval_faithfulness_strict.py
```

- **citation_validity_rate**: citations point to valid evidence_ids
- **supported_claim_rate**: citations whose claim is supported by cited evidence (keyword/number overlap)
- **avg_support_score**: average overlap score among supported claims
- **unsupported_claims**: debug list of claims that look unsupported

Outputs:
- `reports/faithfulness_strict.json`
- `reports/faithfulness_strict_summary.json`

### 4) Leaderboard (CSV for screenshots)
Run after 1–3:
```bash
python scripts/make_leaderboard.py
```

Outputs:
- `reports/leaderboard.csv`
- `reports/leaderboard_summary.txt`

## Optional: LangChain & LangGraph prototypes

This repo is intentionally **framework-light**: the core RAG pipeline (index → retrieve → generate → evaluate) is plain Python so it’s easy to read and debug.

If you want to demonstrate familiarity with common LLM orchestration tools, install the optional dependencies:

```bash
pip install -r requirements_langchain.txt
```

Then run the prototypes (they reuse your existing FAISS index in `data/processed/`):

### LangChain (Runnable pipeline)

```bash
python -m prototypes.langchain_rag_prototype --question "Is metformin appropriate if eGFR is 35?"
```

### LangGraph (StateGraph)

```bash
python -m prototypes.langgraph_rag_prototype --question "Is metformin appropriate if eGFR is 35?"
```

These prototypes use the **offline stub** generator by default (no API key required). You can later wire in an API-backed model (or a local model) inside `src/rag/generate.py` without changing the retrieval logic.

### Example run with OpenAI (optional)
```bash
pip install -r requirements_openai.txt
$env:OPENAI_API_KEY="sk-..."
python -m prototypes.langchain_rag_prototype --question "..." --llm openai --model gpt-4o-mini
```
## Containerization (Docker + Cloud)
This repo can be containerized for reproducible runs locally and in the cloud.

### Build the image
```bash
docker build -t clinrag:latest .
```

### Run the CLI inside the container (offline, free)
Mount `./data` so indices/corpora persist outside the container:
```bash
docker run --rm -v %cd%/data:/app/data clinrag:latest       python -m scripts.rag_cli --question "Is metformin appropriate if eGFR is 35?"
```

### Run the HTTP API (recommended for Cloud Run/ECS)
```bash
docker run --rm -p 8080:8080 -v %cd%/data:/app/data clinrag:latest       uvicorn app.api:app --host 0.0.0.0 --port 8080
```

Then call it:
```bash
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json"       -d "{\"question\": \"Is metformin appropriate if eGFR is 35?\", \"llm\": \"offline\", \"k\": 5}"
```

### OpenAI (optional)
OpenAI usage requires API billing/quota and an API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=%OPENAI_API_KEY% -v %cd%/data:/app/data clinrag:latest       uvicorn app.api:app --host 0.0.0.0 --port 8080
```
(You can still run fully offline by default.)

See `docs/cloud_container.md` for cloud deployment notes.

# 🚀 ClinRAG Healthcare — Azure CI + One‑Command Deploy Guide

This guide sets up:

✅ GitHub Actions → automatically build & push Docker image to Azure Container Registry (ACR)  
✅ One‑command deploy → create infra + deploy Container App  

After setup, deployment becomes:

push → auto build → deploy → live API

---

# ✅ Quick Setup Checklist (First‑time only)

Follow these steps once. After this, everything runs automatically.

---

## A) Add required files into your repo

Unzip the provided bundle and copy these into your project root:

.github/workflows/azure-acr-build.yml  
scripts/deploy_azure.sh  
docs/CI_AND_AZURE_DEPLOY.md  

Your repo should look like:

clinrag-healthcare/  
├── .github/workflows/azure-acr-build.yml  
├── scripts/deploy_azure.sh  
├── docs/CI_AND_AZURE_DEPLOY.md  
├── Dockerfile  
└── ...

---

## B) Create Azure Service Principal (for GitHub Actions)

This allows GitHub to securely push Docker images to Azure.

Run:

az ad sp create-for-rbac --name "clinrag-gh-actions" --role contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RG> \
  --sdk-auth

You will receive JSON output like:

{
  "clientId": "...",
  "clientSecret": "...",
  "subscriptionId": "...",
  "tenantId": "..."
}

Copy this entire JSON block.

---

## C) Add GitHub Secrets + Variables

Go to:

GitHub → Settings → Secrets and variables → Actions

Add:

Secret  
AZURE_CREDENTIALS = JSON from Step B

Variables  
ACR_NAME = your registry name (example: acrclinrag12345)  
IMAGE_REPO = clinrag

---

# 🎉 What happens now?

Every push to main:

git push origin main

GitHub automatically:

1. Builds Docker image  
2. Pushes to Azure Container Registry  
3. (optionally deploys Container App if enabled)

---

# 🚀 One‑Command Azure Deploy

This script:

• creates resource group  
• creates ACR  
• builds image  
• deploys Azure Container App  
• prints live URL  

Run from repo root:

bash scripts/deploy_azure.sh

---

## Optional overrides

RG=rg-clinrag LOC=westus2 APP=clinrag-api bash scripts/deploy_azure.sh

---

# ✅ Test your API

After deploy you’ll get:

https://something.azurecontainerapps.io

Test:

curl https://<URL>/health

curl -X POST https://<URL>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Is metformin appropriate if eGFR is 35?","llm":"offline","k":5}'

---

# 🧠 Pro tips

Local development  
Use Docker mount:

docker run -v ./data:/app/data clinrag:latest

Azure demo  
Bake FAISS index into the image

Production  
Store FAISS index in Azure Blob Storage and download at startup

## Index persistence: Local vs Cloud

### Local development (recommended)
Mount `./data` into the container so indices persist between runs:

```bash
docker build -t clinrag:latest .
docker run --rm -p 8080:8080 -v ./data:/app/data clinrag:latest \
  uvicorn app.api:app --host 0.0.0.0 --port 8080

## Docker (baked PubMedQA index)

This repo bakes a **public + reproducible** retrieval index into the Docker image at build time using **PubMedQA**.

Build + run:

```bash
docker build -t clinrag:latest .
docker run --rm -p 8080:8080 clinrag:latest
```

Test:

```bash
curl -s http://localhost:8080/health
curl -s -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Is metformin appropriate if eGFR is 35?","llm":"offline","k":5}'
```

Notes:
- Image will be larger because it includes the FAISS index + chunk metadata.
- To update corpus/index: rebuild image and redeploy.

