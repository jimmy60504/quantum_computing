#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-space-repo-id> [revision]" >&2
  echo "Example: $0 jimmy60504/Hybrid-QNN-Explorer main" >&2
  exit 1
fi

REPO_ID="$1"
REVISION="${2:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HF_CLI_VENV="${HF_CLI_VENV:-${HOME}/.venvs/hf-cli}"
HF_BIN="${HF_CLI_VENV}/bin/hf"
SPACE_OUTPUT_DIR="${HF_SPACE_OUTPUT_DIR:-${REPO_ROOT}/.out/hf_space_hw1_problem3_publish}"

if [[ ! -x "${HF_BIN}" ]]; then
  python3 -m venv "${HF_CLI_VENV}"
  "${HF_CLI_VENV}/bin/pip" install "huggingface_hub"
fi

"${REPO_ROOT}/HW1/problem3/scripts/gx10_prepare_hf_space.sh"

"${HF_BIN}" repos create "${REPO_ID}" --repo-type space --space-sdk static --public --exist-ok >/dev/null

"${HF_BIN}" upload \
  "${REPO_ID}" \
  "${SPACE_OUTPUT_DIR}" \
  "." \
  --repo-type space \
  --revision "${REVISION}" \
  --exclude ".cache/**" \
  --commit-message "${HF_SPACE_COMMIT_MESSAGE:-Update Hybrid QNN Explorer}"

echo "Uploaded Space bundle to: ${REPO_ID}@${REVISION}"
echo "Space URL: https://huggingface.co/spaces/${REPO_ID}"
