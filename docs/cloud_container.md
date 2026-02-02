# Cloud container notes (quick start)

This repo ships as a standard Docker container (CPU-only). You can run it:
- locally with Docker / docker-compose
- on any container platform (AWS ECS/Fargate, GCP Cloud Run, Azure Container Apps, etc.)

## Local: build + run CLI
```bash
docker build -t clinrag:latest .
docker run --rm -v %cd%/data:/app/data clinrag:latest python -m scripts.rag_cli --question "Is metformin appropriate if eGFR is 35?"
```

## Local: run the HTTP API
```bash
docker build -t clinrag:latest .
docker run --rm -p 8080:8080 -v %cd%/data:/app/data clinrag:latest \
  uvicorn app.api:app --host 0.0.0.0 --port 8080
```

Then:
```bash
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d "{\"question\": \"Is metformin appropriate if eGFR is 35?\", \"llm\": \"offline\", \"k\": 5}"
```

## GCP Cloud Run (example)
1) Build & push to Artifact Registry (or Docker Hub)
2) Deploy to Cloud Run with port 8080

Environment variables:
- `OPENAI_API_KEY` (only if you use `llm=openai`)

Note: For regulated data, keep the default `llm=offline` or use an approved in-VPC model.
