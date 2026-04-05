#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${GX10_HOST:-gx10}"
REMOTE_REPO_DIR="${GX10_REPO_DIR:-/home/jimmy/quantum_computing}"

rsync -av --delete \
  --exclude '.git' \
  --exclude '.DS_Store' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.out' \
  --exclude 'mlruns' \
  --exclude 'mlartifacts' \
  --exclude 'mlflow.db' \
  --exclude '.out/mlflow-postgres-data' \
  --exclude 'HW1/artifacts' \
  --exclude 'logs' \
  --exclude 'hf_space_hw1_problem1/runtime' \
  "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_REPO_DIR}/"
