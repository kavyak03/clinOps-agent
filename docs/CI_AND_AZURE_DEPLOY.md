# CI + One-command Azure deploy additions

## 1) GitHub Actions: auto-build & push to ACR

This workflow builds the Docker image on every push to `main` and pushes it to Azure Container Registry via **ACR Cloud Build**.

### Setup in Azure (Service Principal)
Create a service principal scoped to your resource group and output `--sdk-auth` JSON:

```bash
az ad sp create-for-rbac --name "clinrag-gh-actions" --role contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RG> \
  --sdk-auth
```

In GitHub repo settings:
- **Secrets**: add `AZURE_CREDENTIALS` = the JSON output above
- **Variables**: add
  - `ACR_NAME` = your ACR name (e.g. `acrclinrag12345`)
  - `IMAGE_REPO` = your image repo name (e.g. `clinrag`)

Workflow file:
- `.github/workflows/azure-acr-build.yml`

## 2) One-command deploy script

Run from repo root:

```bash
bash scripts/deploy_azure.sh
```

Optional overrides:

```bash
RG=rg-clinrag LOC=westus2 APP=clinrag-api bash scripts/deploy_azure.sh
```

Notes:
- This script uses `az acr build` (cloud build) and deploys to Azure Container Apps.
- For demo purposes, it uses `latest` tag.
