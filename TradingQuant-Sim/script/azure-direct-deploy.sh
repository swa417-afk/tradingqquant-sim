#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO. Last command: $BASH_COMMAND" >&2' ERR

# -------------------------
# TradingQ Azure Deployment (DevOps friendly, non-interactive)
# -------------------------

# Config (override via env)
RG="${RG:-rg-tradingq-quant}"
LOC="${LOC:-eastus}"
ENV_NAME="${ENV_NAME:-cae-tradingq-quant}"

# App/container settings
APP_NAME="${APP_NAME:-tradingq-api}"
WORKER_NAME="${WORKER_NAME:-tradingq-worker}"
PORT="${PORT:-8000}"

# Image settings
IMAGE_TAG="${IMAGE_TAG:-latest}"
SUFFIX="${SUFFIX:-$(date +%y%m%d%H%M)}"
ACR_NAME="${ACR_NAME:-acrtradingq${SUFFIX}}"     # must be 5-50 chars lowercase/digits only
REPO_API="${REPO_API:-tradingq/api}"
REPO_WORKER="${REPO_WORKER:-tradingq/worker}"

# Build source (choose one)
# - If TARBALL_PATH exists, it will import that image tar
# - Otherwise, it will build from BUILD_CONTEXT paths
TARBALL_PATH="${TARBALL_PATH:-/mnt/data/tradingq-test-build.tar}"
BUILD_CONTEXT_API="${BUILD_CONTEXT_API:-./api}"
BUILD_CONTEXT_WORKER="${BUILD_CONTEXT_WORKER:-./worker}"

# Runtime
CPU="${CPU:-0.5}"
MEM="${MEM:-1.0Gi}"
MIN_REPLICAS="${MIN_REPLICAS:-0}"
MAX_REPLICAS="${MAX_REPLICAS:-1}"

# -------------------------
# Helpers
# -------------------------
info()  { echo -e "[INFO] $*"; }
ok()    { echo -e "[ OK ] $*"; }
warn()  { echo -e "[WARN] $*"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing dependency: $1" >&2; exit 1; }
}

# -------------------------
# Preflight
# -------------------------
need az

info "Using RG=$RG LOC=$LOC ENV=$ENV_NAME"
az account show >/dev/null

# Provider registrations (safe to re-run)
info "Ensuring providers are registered…"
az provider register --namespace Microsoft.App >/dev/null
az provider register --namespace Microsoft.OperationalInsights >/dev/null
az provider register --namespace Microsoft.ContainerRegistry >/dev/null

# -------------------------
# Resource Group
# -------------------------
info "Creating RG (if needed)…"
az group create -n "$RG" -l "$LOC" >/dev/null
ok "RG ready"

# -------------------------
# Log Analytics (recommended for ACA)
# -------------------------
LAW_NAME="${LAW_NAME:-law-tradingq-${SUFFIX}}"
info "Creating Log Analytics Workspace (if needed)…"
az monitor log-analytics workspace create -g "$RG" -n "$LAW_NAME" -l "$LOC" >/dev/null

LAW_ID="$(az monitor log-analytics workspace show -g "$RG" -n "$LAW_NAME" --query customerId -o tsv)"
LAW_KEY="$(az monitor log-analytics workspace get-shared-keys -g "$RG" -n "$LAW_NAME" --query primarySharedKey -o tsv)"
ok "Log Analytics ready"

# -------------------------
# Container Apps Environment
# -------------------------
info "Creating Container Apps Environment (if needed)…"
az containerapp env create \
  -g "$RG" -n "$ENV_NAME" -l "$LOC" \
  --logs-workspace-id "$LAW_ID" \
  --logs-workspace-key "$LAW_KEY" >/dev/null
ok "ACA environment ready"

# -------------------------
# ACR
# -------------------------
info "Creating ACR (if needed)…"
az acr create -g "$RG" -n "$ACR_NAME" --sku Basic >/dev/null
ok "ACR ready: $ACR_NAME"

info "Logging into ACR…"
az acr login -n "$ACR_NAME" >/dev/null

ACR_LOGIN_SERVER="$(az acr show -n "$ACR_NAME" -g "$RG" --query loginServer -o tsv)"

# -------------------------
# Build/import images
# -------------------------
API_IMAGE="${ACR_LOGIN_SERVER}/${REPO_API}:${IMAGE_TAG}"
WORKER_IMAGE="${ACR_LOGIN_SERVER}/${REPO_WORKER}:${IMAGE_TAG}"

if [[ -f "$TARBALL_PATH" ]]; then
  warn "Found tarball at $TARBALL_PATH — importing it into ACR."
  warn "NOTE: ACR import from tar requires a local Docker load/push."

  need docker

  info "Loading tarball into local Docker…"
  docker load -i "$TARBALL_PATH"

  info "Listing loaded images:"
  docker images | head -n 20

  warn "The tarball image names are unknown. Provide the source tags via env to keep it DevOps-friendly."
  : "${API_SRC:?Set API_SRC to the source image tag for the API (e.g. repo:tag).}"
  : "${WORKER_SRC:?Set WORKER_SRC to the source image tag for the worker (e.g. repo:tag).}"

  info "Retagging…"
  docker tag "$API_SRC" "$API_IMAGE"
  docker tag "$WORKER_SRC" "$WORKER_IMAGE"

  info "Pushing to ACR…"
  docker push "$API_IMAGE" >/dev/null
  docker push "$WORKER_IMAGE" >/dev/null
  ok "Images pushed: $API_IMAGE and $WORKER_IMAGE"
else
  warn "Tarball not found. Building from contexts: $BUILD_CONTEXT_API and $BUILD_CONTEXT_WORKER"

  info "Building API image with ACR build…"
  az acr build -r "$ACR_NAME" -t "${REPO_API}:${IMAGE_TAG}" "$BUILD_CONTEXT_API" >/dev/null

  info "Building WORKER image with ACR build…"
  az acr build -r "$ACR_NAME" -t "${REPO_WORKER}:${IMAGE_TAG}" "$BUILD_CONTEXT_WORKER" >/dev/null

  ok "Images built in ACR"
fi

# -------------------------
# Grant ACA pull access to ACR
# -------------------------
info "Ensuring Container Apps can pull from ACR…"
ACR_ID="$(az acr show -g "$RG" -n "$ACR_NAME" --query id -o tsv)"

# Create a user-assigned identity for apps (cleaner than admin creds)
IDENTITY_NAME="${IDENTITY_NAME:-id-tradingq-${SUFFIX}}"
az identity create -g "$RG" -n "$IDENTITY_NAME" >/dev/null
IDENTITY_ID="$(az identity show -g "$RG" -n "$IDENTITY_NAME" --query id -o tsv)"
IDENTITY_PRINCIPAL="$(az identity show -g "$RG" -n "$IDENTITY_NAME" --query principalId -o tsv)"

az role assignment create \
  --assignee-object-id "$IDENTITY_PRINCIPAL" \
  --assignee-principal-type ServicePrincipal \
  --role "AcrPull" \
  --scope "$ACR_ID" >/dev/null || true
ok "Identity + AcrPull ready"

# -------------------------
# Deploy API app (ingress external)
# -------------------------
info "Deploying API app: $APP_NAME"
az containerapp create \
  -g "$RG" -n "$APP_NAME" \
  --environment "$ENV_NAME" \
  --image "$API_IMAGE" \
  --target-port "$PORT" \
  --ingress external \
  --cpu "$CPU" --memory "$MEM" \
  --min-replicas "$MIN_REPLICAS" --max-replicas "$MAX_REPLICAS" \
  --user-assigned "$IDENTITY_ID" \
  --registry-server "$ACR_LOGIN_SERVER" >/dev/null

# -------------------------
# Deploy WORKER app (no ingress)
# -------------------------
info "Deploying WORKER app: $WORKER_NAME"
az containerapp create \
  -g "$RG" -n "$WORKER_NAME" \
  --environment "$ENV_NAME" \
  --image "$WORKER_IMAGE" \
  --ingress internal \
  --cpu "$CPU" --memory "$MEM" \
  --min-replicas "$MIN_REPLICAS" --max-replicas "$MAX_REPLICAS" \
  --user-assigned "$IDENTITY_ID" \
  --registry-server "$ACR_LOGIN_SERVER" >/dev/null

# -------------------------
# Output endpoint
# -------------------------
FQDN="$(az containerapp show -n "$APP_NAME" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
ok "All services deployed!"
info "API URL: https://${FQDN}"
info "Logs: az containerapp logs show -n $APP_NAME -g $RG --follow"
info "Worker logs: az containerapp logs show -n $WORKER_NAME -g $RG --follow"

# -------------------------
# Bicep (optional) - reference in infra/main.bicep
# -------------------------
info "Bicep template available at script/infra/main.bicep for DevOps deployments."
