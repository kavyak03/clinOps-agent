#!/usr/bin/env bash
set -euo pipefail

# One-command Azure deploy for ClinRAG:
# - creates Resource Group
# - creates ACR
# - builds & pushes image via ACR cloud build
# - creates Container Apps environment
# - deploys Container App (public ingress on 8080)
#
# Requirements:
# - Azure CLI installed
# - az login completed
#
# Usage:
#   bash scripts/deploy_azure.sh
#
# Optional env vars (override defaults):
#   RG, LOC, ACR, APPENV, APP, IMAGE_REPO, TARGET_PORT

RG="${RG:-rg-clinrag}"
LOC="${LOC:-westus2}"
ACR="${ACR:-acrclinrag$RANDOM}"          # must be globally unique
APPENV="${APPENV:-clinrag-env}"
APP="${APP:-clinrag-api}"
IMAGE_REPO="${IMAGE_REPO:-clinrag}"
TARGET_PORT="${TARGET_PORT:-8080}"

echo "==> Using:"
echo "  RG=$RG"
echo "  LOC=$LOC"
echo "  ACR=$ACR"
echo "  APPENV=$APPENV"
echo "  APP=$APP"
echo "  IMAGE_REPO=$IMAGE_REPO"
echo "  TARGET_PORT=$TARGET_PORT"

echo "==> Create Resource Group"
az group create -n "$RG" -l "$LOC" 1>/dev/null

echo "==> Create ACR (Basic)"
az acr create -n "$ACR" -g "$RG" --sku Basic 1>/dev/null

echo "==> Build & push image via ACR cloud build"
az acr build -r "$ACR" -t "${IMAGE_REPO}:latest" .

echo "==> Note: FAISS index is baked into the image during Docker build (no runtime mounts needed)."

echo "==> Ensure Container Apps extension"
az extension add --name containerapp --upgrade 1>/dev/null || true

echo "==> Create Container Apps environment (if not exists)"
az containerapp env create -n "$APPENV" -g "$RG" -l "$LOC" 1>/dev/null || true

echo "==> Deploy Container App (public ingress)"
IMG="${ACR}.azurecr.io/${IMAGE_REPO}:latest"

FQDN=$(az containerapp create \
  -n "$APP" -g "$RG" \
  --environment "$APPENV" \
  --image "$IMG" \
  --target-port "$TARGET_PORT" \
  --ingress external \
  --registry-server "${ACR}.azurecr.io" \
  --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "✅ Deployed!"
echo "URL: https://${FQDN}"
echo ""
echo "Test:"
echo "  curl -s https://${FQDN}/health"
echo "  curl -s -X POST https://${FQDN}/ask \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"question\":\"Is metformin appropriate if eGFR is 35?\",\"llm\":\"offline\",\"k\":5}'"
