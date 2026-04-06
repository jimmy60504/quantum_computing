#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-dataset-repo-id> [revision]" >&2
  echo "Example: $0 jimmy60504/data-reuploading-demo-runtime main" >&2
  echo "Optional env: HF_RUNTIME_DATASET_INCLUDE_RUN_IDS=run1,run2" >&2
  exit 1
fi

REPO_ID="$1"
REVISION="${2:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HF_CLI_VENV="${HF_CLI_VENV:-${HOME}/.venvs/hf-cli}"
HF_BIN="${HF_CLI_VENV}/bin/hf"
DATASET_OUTPUT_DIR="${HF_RUNTIME_DATASET_OUTPUT_DIR:-${REPO_ROOT}/.out/hf_space_hw1_runtime_dataset}"

if [[ ! -x "${HF_BIN}" ]]; then
  python3 -m venv "${HF_CLI_VENV}"
  "${HF_CLI_VENV}/bin/pip" install "huggingface_hub"
fi

"${REPO_ROOT}/HW1/problem1/scripts/gx10_prepare_hf_runtime_dataset.sh"

"${HF_BIN}" repos create "${REPO_ID}" --repo-type dataset --public --exist-ok
"${HF_BIN}" upload-large-folder "${REPO_ID}" "${DATASET_OUTPUT_DIR}" --repo-type dataset --revision "${REVISION}" --no-bars

echo "Uploaded runtime dataset bundle to: ${REPO_ID}@${REVISION}"
echo "Runtime root URL: https://huggingface.co/datasets/${REPO_ID}/resolve/${REVISION}/"
if [[ -n "${HF_RUNTIME_DATASET_INCLUDE_RUN_IDS:-}" ]]; then
  echo "Included runs: ${HF_RUNTIME_DATASET_INCLUDE_RUN_IDS}"
fi
