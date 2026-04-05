#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"
GX10_WORKDIR="${GX10_WORKDIR:-/workspace}"
GX10_IMAGE="${GX10_IMAGE:-quantum-gx10:test}"
GX10_LIGHT_CPUSET_DEFAULT="${GX10_LIGHT_CPUSET_DEFAULT:-0-4,10-14}"
GX10_LIGHT_CPUS_DEFAULT="${GX10_LIGHT_CPUS_DEFAULT:-4}"
CPUSET="${GX10_CPUSET:-${GX10_LIGHT_CPUSET_DEFAULT}}"
CPU_COUNT="${GX10_CPUS:-${GX10_LIGHT_CPUS_DEFAULT}}"

exec docker run --rm \
  --cpuset-cpus "${CPUSET}" \
  --cpus "${CPU_COUNT}" \
  --user "$(id -u):$(id -g)" \
  -p "${MLFLOW_PORT}:${MLFLOW_PORT}" \
  -e OMP_NUM_THREADS="${CPU_COUNT}" \
  -e OPENBLAS_NUM_THREADS="${CPU_COUNT}" \
  -e MKL_NUM_THREADS="${CPU_COUNT}" \
  -e NUMEXPR_NUM_THREADS="${CPU_COUNT}" \
  -e VECLIB_MAXIMUM_THREADS="${CPU_COUNT}" \
  -v "${REPO_ROOT}:${GX10_WORKDIR}" \
  -w "${GX10_WORKDIR}" \
  "${GX10_IMAGE}" \
  mlflow ui \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}"
