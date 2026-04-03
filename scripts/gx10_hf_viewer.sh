#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_VIEWER_PORT="${HF_VIEWER_PORT:-7860}"

cd "${REPO_ROOT}/hf_space_hw1"
exec python3 -m http.server "${HF_VIEWER_PORT}"
