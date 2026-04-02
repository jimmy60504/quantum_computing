#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
GX10_WORKDIR="${GX10_WORKDIR:-/workspace}"
GX10_IMAGE="${GX10_IMAGE:-quantum-gx10:test}"

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -p "${MLFLOW_PORT}:${MLFLOW_PORT}" \
  -v "${REPO_ROOT}:${GX10_WORKDIR}" \
  -w "${GX10_WORKDIR}" \
  "${GX10_IMAGE}" \
  mlflow ui \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
