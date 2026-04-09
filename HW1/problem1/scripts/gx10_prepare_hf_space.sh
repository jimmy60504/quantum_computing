#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/HW1/problem1/hf_space"
RUNTIME_DIR="${SOURCE_DIR}/runtime"
OUTPUT_DIR="${HF_SPACE_OUTPUT_DIR:-${REPO_ROOT}/.out/hw1_problem1_hf_space_publish}"
SOURCE_EXPORT_DIR="${OUTPUT_DIR}/source"
RUNTIME_DATASET_REPO="${HF_SPACE_RUNTIME_DATASET_REPO:-}"
RUNTIME_DATASET_REVISION="${HF_SPACE_RUNTIME_DATASET_REVISION:-main}"

if [[ -n "${HF_SPACE_INCLUDE_RUNTIME:-}" ]]; then
  INCLUDE_RUNTIME="${HF_SPACE_INCLUDE_RUNTIME}"
else
  INCLUDE_RUNTIME="0"
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cp -R "${SOURCE_DIR}/." "${OUTPUT_DIR}/"
rm -rf "${OUTPUT_DIR}/runtime"

if [[ "${INCLUDE_RUNTIME}" == "1" && -d "${RUNTIME_DIR}" ]]; then
  mkdir -p "${OUTPUT_DIR}/runtime"
  rsync -av \
    --exclude 'metrics/' \
    --exclude '*_snapshots.json' \
    --exclude 'chunks/*_snapshots_chunk_*.json' \
    --exclude 'metrics/*' \
    "${RUNTIME_DIR}/" "${OUTPUT_DIR}/runtime/"
fi

if [[ -n "${RUNTIME_DATASET_REPO}" ]]; then
  RUNTIME_ROOT_URL="https://huggingface.co/datasets/${RUNTIME_DATASET_REPO}/resolve/${RUNTIME_DATASET_REVISION}/"
  mkdir -p "${OUTPUT_DIR}/data"
  cat > "${OUTPUT_DIR}/data/runtime_source.json" <<EOF
{
  "runtime_root_url": "${RUNTIME_ROOT_URL}",
  "manifest_path": "runtime/viewer_manifest.json",
  "fallback_manifest_urls": [
    "./data/viewer_manifest.template.json"
  ]
}
EOF
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
  --exclude 'hf_space/runtime/' \
  "${REPO_ROOT}/HW1/problem1/" \
  "${SOURCE_EXPORT_DIR}/HW1/problem1/"

rsync -av \
  "${REPO_ROOT}/scripts/gx10_run_py.sh" \
  "${SOURCE_EXPORT_DIR}/scripts/"

echo "Prepared HF Space bundle at: ${OUTPUT_DIR}"
echo "include_runtime=${INCLUDE_RUNTIME}"
if [[ -n "${RUNTIME_DATASET_REPO}" ]]; then
  echo "runtime_dataset_repo=${RUNTIME_DATASET_REPO}"
  echo "runtime_dataset_revision=${RUNTIME_DATASET_REVISION}"
fi
echo "Next: create a Hugging Face Static Space repo and push the contents of this directory."
