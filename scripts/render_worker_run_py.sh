#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <python-script> [args...]" >&2
  exit 1
fi

IMAGE_TAG="${RENDER_IMAGE_TAG:-quantum-render-worker:x86_64}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE_TAG}" \
  python "$@"
