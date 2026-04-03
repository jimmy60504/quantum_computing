#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_VIEWER_PORT="${HF_VIEWER_PORT:-7860}"

existing_pids="$(lsof -tiTCP:${HF_VIEWER_PORT} -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "${existing_pids}" ]]; then
  echo "Stopping existing viewer on port ${HF_VIEWER_PORT}: ${existing_pids}"
  kill ${existing_pids} 2>/dev/null || true
  sleep 1

  remaining_pids="$(lsof -tiTCP:${HF_VIEWER_PORT} -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${remaining_pids}" ]]; then
    echo "Force stopping viewer on port ${HF_VIEWER_PORT}: ${remaining_pids}"
    kill -9 ${remaining_pids} 2>/dev/null || true
  fi
fi

cd "${REPO_ROOT}/hf_space_hw1_problem1"
exec python3 -m http.server "${HF_VIEWER_PORT}"
