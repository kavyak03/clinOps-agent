# Generic CI/CD Runbook

This repo includes a **generic CI/CD setup** that does not require AWS, Azure, GCP, or any external cloud container registry.

## What this setup gives you

1. **CI on every push / PR**
   - install dependencies
   - compile Python source
   - build the Docker image

2. **Packaged Docker image artifact on tags or manual runs**
   - builds the image in GitHub Actions
   - exports it as a `.tar.gz`
   - uploads it as a downloadable GitHub Actions artifact

3. **Optional deployment to any self-hosted Docker machine**
   - uses a GitHub self-hosted runner
   - no cloud credentials required
   - supports GitHub `staging` and `prod` environments

---

## Files added

- `.github/workflows/ci.yml`
- `.github/workflows/package-image.yml`
- `.github/workflows/deploy-selfhosted.yml`
- `infra/docker-deploy/compose.selfhosted.yml`
- `infra/docker-deploy/.env.template`

---

## Recommended usage modes

### Mode A — CI only (simplest)
Use this if you only want:
- automatic checks on pull requests
- Docker image build validation

Workflow:
- push branch / open PR
- GitHub Actions runs `CI`

### Mode B — Package a downloadable image
Use this if you want a build artifact you can download and run anywhere.

Trigger:
- push a tag like `v1.0.0`
- or run `Package Docker Image` manually from GitHub Actions

Output:
- `clinops-agent-<sha>.tar.gz` as a GitHub artifact

To load it on another machine:

```bash
gunzip clinops-agent-<sha>.tar.gz
docker load -i clinops-agent-<sha>.tar
docker images
```

Then run it with your own `.env` and Docker Compose setup.

### Mode C — Self-hosted deployment

Use this if you have:

- any Linux box / VM / desktop that can run Docker
- a GitHub self-hosted runner installed on it

The workflow `Deploy to Self-Hosted Docker Host` builds and starts the service on that machine.

---

## GitHub environments

Create two GitHub environments if you want promotion flow:

- `staging`
- `prod`

In each environment, add these secrets:

- `LLM_PROVIDER`
- `LLM_MODEL`
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `VECTOR_STORE`
- `DATABASE_URL`
- `PGVECTOR_TABLE`
- `RERANK_ENABLE`
- `RERANK_CANDIDATES`
- `RERANK_MODEL`
- `LOG_LEVEL`

You can leave unused provider keys blank.

### Important note about `DATABASE_URL`

For self-hosted deployment, your API container must be able to reach the database specified by `DATABASE_URL`.

Examples:
- external Postgres instance
- local Postgres on the same host
- another Docker network service

If you want the API and DB on the same host, create a matching database service or point to an existing database.

NOTE: The self-hosted deployment workflow is included to demonstrate how the API can be deployed automatically to a Docker host using a GitHub Actions self-hosted runner. This workflow has not been executed in this repository because no dedicated runner host has been provisioned. To enable it, install a GitHub self-hosted runner on a machine with Docker and trigger the Mode C workflow mentioned above.

---

## Local developer workflow stays the same

Nothing changes for local development.

You still use:

```bash
docker compose up --build -d
make ingest
make eval
```
