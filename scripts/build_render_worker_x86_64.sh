#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${RENDER_IMAGE_TAG:-quantum-render-worker:x86_64}"

docker build \
  --platform linux/amd64 \
  -f "${REPO_ROOT}/Dockerfile.render-x86_64" \
  -t "${IMAGE_TAG}" \
  "${REPO_ROOT}"
