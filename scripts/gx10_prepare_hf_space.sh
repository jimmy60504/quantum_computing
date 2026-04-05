#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/hf_space_hw1_problem1"
RUNTIME_DIR="${SOURCE_DIR}/runtime"
OUTPUT_DIR="${HF_SPACE_OUTPUT_DIR:-${REPO_ROOT}/.out/hf_space_hw1_problem1_publish}"
SOURCE_EXPORT_DIR="${OUTPUT_DIR}/source"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cp -R "${SOURCE_DIR}/." "${OUTPUT_DIR}/"
rm -rf "${OUTPUT_DIR}/runtime"

if [[ -d "${RUNTIME_DIR}" ]]; then
  mkdir -p "${OUTPUT_DIR}/runtime"
  rsync -av "${RUNTIME_DIR}/" "${OUTPUT_DIR}/runtime/"
fi

mkdir -p "${SOURCE_EXPORT_DIR}"
mkdir -p \
  "${SOURCE_EXPORT_DIR}/docker" \
  "${SOURCE_EXPORT_DIR}/HW1" \
  "${SOURCE_EXPORT_DIR}/scripts"

rsync -av \
  "${REPO_ROOT}/README.md" \
  "${REPO_ROOT}/environment.yml" \
  "${REPO_ROOT}/Dockerfile" \
  "${SOURCE_EXPORT_DIR}/"

rsync -av \
  "${REPO_ROOT}/docker/requirements-gx10.txt" \
  "${SOURCE_EXPORT_DIR}/docker/"

rsync -av \
  "${REPO_ROOT}/HW1/problem1/" \
  "${SOURCE_EXPORT_DIR}/HW1/problem1/"

rsync -av \
  "${REPO_ROOT}/scripts/gx10_run_py.sh" \
  "${REPO_ROOT}/scripts/gx10_hf_viewer.sh" \
  "${REPO_ROOT}/scripts/gx10_prepare_hf_space.sh" \
  "${SOURCE_EXPORT_DIR}/scripts/"

echo "Prepared HF Space bundle at: ${OUTPUT_DIR}"
echo "Next: create a Hugging Face Static Space repo and push the contents of this directory."
