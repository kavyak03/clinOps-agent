SHELL := /bin/bash

.PHONY: help up down logs rebuild ingest eval shell dbshell health

help:
	@echo "Targets:"
	@echo "  make up        - build + start services"
	@echo "  make down      - stop services"
	@echo "  make logs      - follow api logs"
	@echo "  make rebuild   - rebuild api image"
	@echo "  make ingest    - ingest docs from data/raw into pgvector"
	@echo "  make eval      - run eval harness against local API"
	@echo "  make shell     - open a shell in api container"
	@echo "  make dbshell   - open psql in db container"
	@echo "  make health    - call /healthz"

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

rebuild:
	docker compose build --no-cache api

ingest:
	docker compose exec api sh -lc 'PYTHONPATH=/app python scripts/ingest.py --input data/raw --glob "**/*.*" --chunk_chars 1200 --overlap 200'

eval:
	EVAL_API_BASE=http://localhost:8000 python -m src.eval.run_eval

shell:
	docker compose exec api bash

dbshell:
	docker compose exec db psql -U clinops -d clinops

health:
	curl -s http://localhost:8000/healthz || true


run-sim:
	docker compose exec api python scripts/run_simulation.py

run-validate:
	docker compose exec api python scripts/run_validation.py

run-decision:
	docker compose exec api python scripts/run_decision_demo.py

run-pipeline:
	docker compose exec api python scripts/run_pipeline.py --api-base http://localhost:8000
