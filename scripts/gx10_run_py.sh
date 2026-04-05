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
GX10_HEAVY_CPUSET_DEFAULT="${GX10_HEAVY_CPUSET_DEFAULT:-5-9,15-19}"
GX10_HEAVY_CPUS_DEFAULT="${GX10_HEAVY_CPUS_DEFAULT:-10}"
CPUSET="${GX10_CPUSET:-${GX10_HEAVY_CPUSET_DEFAULT}}"
CPU_COUNT="${GX10_CPUS:-${GX10_HEAVY_CPUS_DEFAULT}}"

PYTHON_FILE="$1"
shift

if [[ ! -f "${REPO_ROOT}/${PYTHON_FILE}" ]]; then
  echo "Python file not found: ${PYTHON_FILE}" >&2
  exit 1
fi

docker run --rm --gpus all \
  --cpuset-cpus "${CPUSET}" \
  --cpus "${CPU_COUNT}" \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e OMP_NUM_THREADS="${CPU_COUNT}" \
  -e OPENBLAS_NUM_THREADS="${CPU_COUNT}" \
  -e MKL_NUM_THREADS="${CPU_COUNT}" \
  -e NUMEXPR_NUM_THREADS="${CPU_COUNT}" \
  -e VECLIB_MAXIMUM_THREADS="${CPU_COUNT}" \
  -v "${REPO_ROOT}:${WORKDIR}" \
  -w "${WORKDIR}" \
  "${IMAGE}" \
  python "${PYTHON_FILE}" "$@"
