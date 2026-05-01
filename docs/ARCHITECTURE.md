# Unified Biomedical AI Validation Architecture

## Narrative
This repository is designed as a single flagship system:

- structured biomedical data
- baseline risk modeling
- RAG + agentic LLM reasoning
- validation gates
- decision-grade output

## New layers
- `src/simulation`: synthetic cohort generation and perturbation
- `src/models`: simple baseline risk scoring
- `src/validation`: validation gates
- `src/decision`: structured decision output

## Demo entrypoints
- `scripts/run_simulation.py`
- `scripts/run_validation.py`
- `scripts/run_decision_demo.py`
- `scripts/run_pipeline.py`

## API
- `/ask` : RAG
- `/agent/ask` : agentic RAG
- `/decision/ask` : validated decision output
