#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/hf_space_hw1"
RUNTIME_DIR="${SOURCE_DIR}/runtime"
OUTPUT_DIR="${HF_SPACE_OUTPUT_DIR:-${REPO_ROOT}/.out/hf_space_hw1_publish}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

rsync -av \
  --exclude 'runtime' \
  "${SOURCE_DIR}/" "${OUTPUT_DIR}/"

if [[ -d "${RUNTIME_DIR}" ]]; then
  mkdir -p "${OUTPUT_DIR}/runtime"
  rsync -av "${RUNTIME_DIR}/" "${OUTPUT_DIR}/runtime/"
fi

echo "Prepared HF Space bundle at: ${OUTPUT_DIR}"
echo "Next: create a Hugging Face Static Space repo and push the contents of this directory."
