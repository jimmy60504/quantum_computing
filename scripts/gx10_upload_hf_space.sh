#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-space-repo-id> [revision]" >&2
  echo "Example: $0 jimmy60504/Data-Reuploading-Demo main" >&2
  echo "Optional env: HF_SPACE_RUNTIME_DATASET_REPO=jimmy60504/data-reuploading-demo-runtime-test" >&2
  exit 1
fi

REPO_ID="$1"
REVISION="${2:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HF_CLI_VENV="${HF_CLI_VENV:-${HOME}/.venvs/hf-cli}"
HF_BIN="${HF_CLI_VENV}/bin/hf"
SPACE_OUTPUT_DIR="${HF_SPACE_OUTPUT_DIR:-${REPO_ROOT}/.out/hf_space_hw1_problem1_publish}"
SPACE_COMMIT_MESSAGE="${HF_SPACE_COMMIT_MESSAGE:-Update static Space bundle}"
RUNTIME_DATASET_REPO="${HF_SPACE_RUNTIME_DATASET_REPO:-}"
RUNTIME_DATASET_REVISION="${HF_SPACE_RUNTIME_DATASET_REVISION:-main}"

if [[ ! -x "${HF_BIN}" ]]; then
  python3 -m venv "${HF_CLI_VENV}"
  "${HF_CLI_VENV}/bin/pip" install "huggingface_hub"
fi

if [[ -n "${RUNTIME_DATASET_REPO}" ]]; then
  HF_SPACE_RUNTIME_DATASET_REPO="${RUNTIME_DATASET_REPO}" \
  HF_SPACE_RUNTIME_DATASET_REVISION="${RUNTIME_DATASET_REVISION}" \
  "${REPO_ROOT}/scripts/gx10_prepare_hf_space.sh"
else
  "${REPO_ROOT}/scripts/gx10_prepare_hf_space.sh"
fi

"${HF_BIN}" repos create "${REPO_ID}" --repo-type space --space-sdk static --public --exist-ok >/dev/null

UPLOAD_ARGS=(
  "${HF_BIN}" upload
  "${REPO_ID}"
  "${SPACE_OUTPUT_DIR}"
  "."
  --repo-type space
  --revision "${REVISION}"
  --exclude ".cache/**"
  --commit-message "${SPACE_COMMIT_MESSAGE}"
)

if [[ -n "${RUNTIME_DATASET_REPO}" ]]; then
  UPLOAD_ARGS+=(--delete "runtime/**")
fi

"${UPLOAD_ARGS[@]}"

echo "Uploaded Space bundle to: ${REPO_ID}@${REVISION}"
echo "Space URL: https://huggingface.co/spaces/${REPO_ID}"
if [[ -n "${RUNTIME_DATASET_REPO}" ]]; then
  echo "Runtime dataset: https://huggingface.co/datasets/${RUNTIME_DATASET_REPO}"
fi
