#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {start|stop|restart|status|logs}" >&2
  exit 1
fi

ACTION="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GX10_WORKDIR="${GX10_WORKDIR:-/workspace}"
GX10_IMAGE="${GX10_IMAGE:-quantum-gx10:test}"
MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-*}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${GX10_WORKDIR}/mlartifacts}"
MLFLOW_DB_CONTAINER="${MLFLOW_DB_CONTAINER:-gx10-mlflow-postgres}"
MLFLOW_SERVER_CONTAINER="${MLFLOW_SERVER_CONTAINER:-gx10-mlflow-server}"
MLFLOW_DB_NAME="${MLFLOW_DB_NAME:-mlflow}"
MLFLOW_DB_USER="${MLFLOW_DB_USER:-mlflow}"
MLFLOW_DB_PASSWORD="${MLFLOW_DB_PASSWORD:-mlflow}"
MLFLOW_DB_PORT="${MLFLOW_DB_PORT:-5432}"
MLFLOW_DB_DATA_DIR="${MLFLOW_DB_DATA_DIR:-${REPO_ROOT}/.out/mlflow-postgres-data}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-postgresql+psycopg://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@${MLFLOW_DB_CONTAINER}:${MLFLOW_DB_PORT}/${MLFLOW_DB_NAME}}"

GX10_LIGHT_CPUSET_DEFAULT="${GX10_LIGHT_CPUSET_DEFAULT:-0-4,10-14}"
GX10_LIGHT_CPUS_DEFAULT="${GX10_LIGHT_CPUS_DEFAULT:-4}"
CPUSET="${GX10_CPUSET:-${GX10_LIGHT_CPUSET_DEFAULT}}"
CPU_COUNT="${GX10_CPUS:-${GX10_LIGHT_CPUS_DEFAULT}}"

ensure_network() {
  docker network inspect gx10-mlflow >/dev/null 2>&1 || docker network create gx10-mlflow >/dev/null
}

start_postgres() {
  mkdir -p "${MLFLOW_DB_DATA_DIR}"
  if docker ps -a --format '{{.Names}}' | grep -qx "${MLFLOW_DB_CONTAINER}"; then
    docker start "${MLFLOW_DB_CONTAINER}" >/dev/null
    return
  fi

  docker run -d \
    --name "${MLFLOW_DB_CONTAINER}" \
    --network gx10-mlflow \
    --cpuset-cpus "${CPUSET}" \
    --cpus "${CPU_COUNT}" \
    -e POSTGRES_DB="${MLFLOW_DB_NAME}" \
    -e POSTGRES_USER="${MLFLOW_DB_USER}" \
    -e POSTGRES_PASSWORD="${MLFLOW_DB_PASSWORD}" \
    -v "${MLFLOW_DB_DATA_DIR}:/var/lib/postgresql/data" \
    postgres:14.18 >/dev/null
}

wait_for_postgres() {
  local attempts=60
  until docker exec "${MLFLOW_DB_CONTAINER}" pg_isready -U "${MLFLOW_DB_USER}" -d "${MLFLOW_DB_NAME}" >/dev/null 2>&1; do
    attempts=$((attempts - 1))
    if [[ "${attempts}" -le 0 ]]; then
      echo "Postgres did not become ready in time." >&2
      exit 1
    fi
    sleep 1
  done
}

start_mlflow_server() {
  mkdir -p "${REPO_ROOT}/mlartifacts"
  if docker ps -a --format '{{.Names}}' | grep -qx "${MLFLOW_SERVER_CONTAINER}"; then
    docker rm -f "${MLFLOW_SERVER_CONTAINER}" >/dev/null 2>&1 || true
  fi

  docker run -d \
    --name "${MLFLOW_SERVER_CONTAINER}" \
    --network gx10-mlflow \
    --cpuset-cpus "${CPUSET}" \
    --cpus "${CPU_COUNT}" \
    -p "${MLFLOW_PORT}:${MLFLOW_PORT}" \
    --user "$(id -u):$(id -g)" \
    -e OMP_NUM_THREADS="${CPU_COUNT}" \
    -e OPENBLAS_NUM_THREADS="${CPU_COUNT}" \
    -e MKL_NUM_THREADS="${CPU_COUNT}" \
    -e NUMEXPR_NUM_THREADS="${CPU_COUNT}" \
    -e VECLIB_MAXIMUM_THREADS="${CPU_COUNT}" \
    -v "${REPO_ROOT}:${GX10_WORKDIR}" \
    -w "${GX10_WORKDIR}" \
    "${GX10_IMAGE}" \
    mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
    --host "${MLFLOW_HOST}" \
    --port "${MLFLOW_PORT}" >/dev/null
}

status() {
  docker ps --filter "name=${MLFLOW_DB_CONTAINER}" --filter "name=${MLFLOW_SERVER_CONTAINER}"
  echo "tracking_uri=http://127.0.0.1:${MLFLOW_PORT}"
  echo "backend_store_uri=${MLFLOW_BACKEND_STORE_URI}"
  echo "allowed_hosts=${MLFLOW_ALLOWED_HOSTS}"
}

case "${ACTION}" in
  start)
    ensure_network
    start_postgres
    wait_for_postgres
    start_mlflow_server
    status
    ;;
  stop)
    docker rm -f "${MLFLOW_SERVER_CONTAINER}" >/dev/null 2>&1 || true
    docker rm -f "${MLFLOW_DB_CONTAINER}" >/dev/null 2>&1 || true
    ;;
  restart)
    "${BASH_SOURCE[0]}" stop
    "${BASH_SOURCE[0]}" start
    ;;
  status)
    status
    ;;
  logs)
    docker logs --tail 200 "${MLFLOW_SERVER_CONTAINER}"
    ;;
  *)
    echo "Unsupported action: ${ACTION}" >&2
    exit 1
    ;;
esac
