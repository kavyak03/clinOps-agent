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
#On Windows powershell
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

