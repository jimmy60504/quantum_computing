#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <repo-relative-python-file> [script args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORKDIR="${GX10_WORKDIR:-/workspace}"
IMAGE="${GX10_IMAGE:-quantum-gx10:test}"

PYTHON_FILE="$1"
shift

if [[ ! -f "${REPO_ROOT}/${PYTHON_FILE}" ]]; then
  echo "Python file not found: ${PYTHON_FILE}" >&2
  exit 1
fi

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${REPO_ROOT}:${WORKDIR}" \
  -w "${WORKDIR}" \
  "${IMAGE}" \
  python "${PYTHON_FILE}" "$@"
