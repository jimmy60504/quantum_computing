#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
./scripts/gx10_mlflow_server.sh start

TRACKING_URI="${PROB2_TRACKING_URI:-http://gx10-mlflow-server:5001}"
EPOCHS="${PROB2_EPOCHS:-50}"
SAMPLES="${PROB2_SAMPLES:-200}"
EXPORT_EVERY="${PROB2_EXPORT_EVERY:-5}"

configs=(
  "4 4"
  "6 6"
  "2 2"
  "4 8"
)

for config in "${configs[@]}"; do
  read -r layers_explicit layers_reuploading <<< "${config}"
  run_name="q2-le${layers_explicit}-lr${layers_reuploading}-e${EPOCHS}"

  echo "[prob2] running ${run_name}"
  GX10_DOCKER_NETWORK=gx10-mlflow \
  ./scripts/gx10_run_py.sh HW1/problem2/train.py \
    --epochs "${EPOCHS}" \
    --n-samples "${SAMPLES}" \
    --layers-explicit "${layers_explicit}" \
    --layers-reuploading "${layers_reuploading}" \
    --viewer-export-every "${EXPORT_EVERY}" \
    --tracking-uri "${TRACKING_URI}" \
    --run-name "${run_name}"
done
