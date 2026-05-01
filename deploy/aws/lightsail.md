# AWS Lightsail Deployment (Docker)

This deploys the ClinOps Agent API + Postgres (pgvector) using Docker on an AWS Lightsail instance.

## Why Lightsail
Fastest path to demonstrate:
- containerized deployment
- environment-variable secrets
- logs
- reproducible infrastructure

## 1) Create a Lightsail instance
- OS: Ubuntu 22.04+
- Choose a small plan for demo; upgrade if needed
- Attach a Static IP

## 2) Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

## 3) Copy repo to server
- Option A: SCP the repo
- Option B: git clone (after you push)

## 4) Configure environment
Create a `.env` file:
```env
DATABASE_URL=postgresql+psycopg2://clinops:clinops@db:5432/clinops
VECTOR_STORE=pgvector
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=YOUR_KEY
LOG_LEVEL=INFO
```

## 5) Run
```bash
docker compose --env-file .env up --build -d
docker compose logs -f api
```

## 6) Open networking
In Lightsail Networking:
- Open TCP port 8000 (or use a reverse proxy for 80/443)

Test:
```bash
curl http://YOUR_STATIC_IP:8000/healthz
```

## HTTPS (optional)
Use Nginx/Caddy + LetsEncrypt to serve HTTPS on 443 and proxy to 8000.
