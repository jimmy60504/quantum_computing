#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/HW1/problem3/hf_space"
RUNTIME_DIR="${SOURCE_DIR}/runtime"
OUTPUT_DIR="${HF_SPACE_OUTPUT_DIR:-${REPO_ROOT}/.out/hf_space_hw1_problem3_publish}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cp -R "${SOURCE_DIR}/." "${OUTPUT_DIR}/"
rm -rf "${OUTPUT_DIR}/runtime"

if [[ -d "${RUNTIME_DIR}" ]]; then
  mkdir -p "${OUTPUT_DIR}/runtime"
  rsync -av \
    --exclude 'logs/' \
    --exclude 'runs/*/checkpoints/' \
    "${RUNTIME_DIR}/" "${OUTPUT_DIR}/runtime/"
fi

echo "Prepared HF Space bundle at: ${OUTPUT_DIR}"
du -sh "${OUTPUT_DIR}"
echo "Next: run gx10_upload_hf_space.sh <repo-id>"
