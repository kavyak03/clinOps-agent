# ClinRAG — Healthcare RAG with Synthetic Data + Public Corpora  
**Bioinformatics × LLM Engineering × Reproducible Deployment**

<!-- =======================
BADGES (replace <...> once your repo path is final)
======================= -->
[![CI](https://github.com/<YOUR_GH_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_GH_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

ClinRAG is a **reproducible healthcare Retrieval-Augmented Generation (RAG) mini-system** showing how to build safer clinical LLM workflows:

- Synthetic patient charts (no PHI)
- Public corpora option (PubMedQA)
- Embeddings + FAISS retrieval
- Structured JSON outputs with citations
- Faithfulness + grounding evaluation harness
- Offline mode (no API calls required)
- Docker + cloud-ready (Azure optional)

---

## 1‑minute quickstart (Docker — recommended)

This is the path most recruiters / reviewers should use.  
✅ No local Python headaches. ✅ Reproducible.

### Build (bakes PubMedQA + FAISS index into the image)
```bash
docker build -t clinrag:latest .
```

### Run API
```bash
docker run --rm -p 8080:8080 clinrag:latest
```

### Test
```bash
curl -s http://localhost:8080/health

curl -s -X POST http://localhost:8080/ask   -H "Content-Type: application/json"   -d "{"question":"Is metformin appropriate if eGFR is 35?","llm":"offline","k":5}"
```

---

## Why this project exists

Healthcare LLM systems must be:

✅ grounded  
✅ auditable  
✅ reproducible  
✅ safe against hallucinations  

This repo demonstrates engineering patterns used in real systems:

- retrieval over approved corpora only
- answer must cite evidence
- strict structured output
- offline fallback mode
- evaluation harness for retrieval + faithfulness

---

## Repository structure

```
app/                  FastAPI service (runtime entrypoint)
src/                  core RAG + QC logic
scripts/              CLIs (data/QC/index/demo/eval)
prototypes/           LangChain + LangGraph demos
data/
  raw/                synthetic patients/notes (generated)
  corpora/             synthetic or public corpora
  processed/           FAISS index + chunk metadata (generated or baked)
reports/              evaluation outputs
.github/workflows/     CI + optional Azure workflows
Dockerfile             reproducible container build
docs/                  extra notes (optional)
```

---

## Demo GIF (optional but recommended)

Add a short 10–20s screen recording of:

1) `docker build ...`
2) `docker run ...`
3) `curl /ask ...`
4) show JSON output

Save it as:

```
docs/demo.gif
```

Then this link will render it:

![ClinRAG demo](docs/demo.gif)

> Tip: On Windows, you can record with Xbox Game Bar; on Mac use QuickTime. Export as GIF via an online converter or `ffmpeg`.

---

## Screenshots (optional but recommended)

Add a couple images for fast “scanability”:

- `docs/screenshots/api_response.png` (example `/ask` response)
- `docs/screenshots/eval_plots.png` (one combined plot collage or a single representative plot)

Embed them:

![API response](docs/screenshots/api_response.png)
![Evaluation plots](docs/screenshots/eval_plots.png)

---

## OpenAI integration (optional)

Offline mode is the default. If you want model-generated answers:

### Set key
**Windows PowerShell**
```powershell
$env:OPENAI_API_KEY="sk-..."
```

**macOS/Linux**
```bash
export OPENAI_API_KEY="sk-..."
```

### Run container with key
```bash
docker run --rm -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY clinrag:latest
```

### Call with `llm=openai`
```bash
curl -s -X POST http://localhost:8080/ask   -H "Content-Type: application/json"   -d "{"question":"Is metformin appropriate if eGFR is 35?","llm":"openai","k":5}"
```

> OpenAI requires API billing/quota. Offline mode always works.

---

## Optional: Local Python run (no Docker)

Use this if you want VS Code native runs. **Python 3.11+ recommended**.

### 1) Create env
**Windows PowerShell**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

**macOS/Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2) Install deps
```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_langchain.txt
pip install -r requirements_openai.txt
```

### 3) Choose data mode

#### A) Synthetic mode (offline)
```bash
python -m scripts.make_data
python -m scripts.run_qc
python -m scripts.build_index
```

#### B) Public PubMedQA mode (public + reproducible)
```bash
python -m scripts.download_public_corpus_pubmedqa --config pqa_labeled --split train --max_examples 2000
python -m scripts.build_index --corpus data/corpora/public/pubmedqa_corpus_singleline.jsonl
```

### 4) Run API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8080
```

---

## Evaluation (offline + deterministic)

### Retrieval evaluation
```bash
python -m scripts.eval_retrieval
```

### Generation heuristics
```bash
python -m scripts.eval_generation_heuristics
```

### Strict faithfulness checks
```bash
python -m scripts.eval_faithfulness_strict
```

### Leaderboard
```bash
python -m scripts.make_leaderboard
```

Outputs go to `reports/`.

---

## LangChain + LangGraph prototypes

Install optional deps:
```bash
pip install -r requirements_langchain.txt
```

### LangChain
```bash
python -m prototypes.langchain_rag_prototype   --question "Is metformin appropriate if eGFR is 35?"
```

### LangGraph
```bash
python -m prototypes.langgraph_rag_prototype   --question "Is metformin appropriate if eGFR is 35?"
```

> If running locally (non-Docker), run `python -m scripts.build_index` first so `data/processed/` exists.

---

## CI (free) + Cloud (optional)

- Free GitHub Actions CI runs on every push (`.github/workflows/ci.yml`)
- Azure build/deploy is optional; keep it **manual-only** until you want a live cloud demo

---

## License

MIT