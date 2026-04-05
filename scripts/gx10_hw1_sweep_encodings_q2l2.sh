#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/hw1_encoding_sweeps"

DEVICE="${HW1_SWEEP_DEVICE:-lightning.qubit}"
DIFF_METHOD="${HW1_SWEEP_DIFF_METHOD:-adjoint}"
EPOCHS="${HW1_SWEEP_EPOCHS:-20}"
BATCH_SIZE="${HW1_SWEEP_BATCH_SIZE:-64}"
NUM_SAMPLES="${HW1_SWEEP_NUM_SAMPLES:-1000}"
HEATMAP_GRID_SIZE="${HW1_SWEEP_HEATMAP_GRID_SIZE:-24}"
RENDER_WORKERS="${HW1_SWEEP_RENDER_WORKERS:-16}"
VIEWER_EXPORT_EVERY="${HW1_SWEEP_VIEWER_EXPORT_EVERY:-1}"
ENCODINGS_CSV="${HW1_SWEEP_ENCODINGS:-raw,poly,exp}"
QUBITS="${HW1_SWEEP_QUBITS:-2}"
LAYERS="${HW1_SWEEP_LAYERS:-2}"

IFS=',' read -r -a ENCODINGS <<< "${ENCODINGS_CSV}"

mkdir -p "${LOG_DIR}"

echo "HW1 encoding sweep starting"
echo "repo=${REPO_ROOT}"
echo "device=${DEVICE} diff_method=${DIFF_METHOD}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_samples=${NUM_SAMPLES}"
echo "heatmap_grid_size=${HEATMAP_GRID_SIZE} render_workers=${RENDER_WORKERS}"
echo "encodings=${ENCODINGS_CSV} qubits=${QUBITS} layers=${LAYERS}"
echo "logs=${LOG_DIR}"
echo

for encoding in "${ENCODINGS[@]}"; do
  RUN_NAME="${encoding}-q${QUBITS}-l${LAYERS}-e${EPOCHS}"
  LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

  echo "============================================================"
  echo "Running ${RUN_NAME}"
  echo "log=${LOG_PATH}"
  echo "============================================================"

  (
    cd "${REPO_ROOT}"
    ./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
      --device "${DEVICE}" \
      --diff-method "${DIFF_METHOD}" \
      --encoding "${encoding}" \
      --num-qubits "${QUBITS}" \
      --num-layers "${LAYERS}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-samples "${NUM_SAMPLES}" \
      --heatmap-grid-size "${HEATMAP_GRID_SIZE}" \
      --render-workers "${RENDER_WORKERS}" \
      --viewer-export-every "${VIEWER_EXPORT_EVERY}" \
      --run-name "${RUN_NAME}"
  ) 2>&1 | tee "${LOG_PATH}"

  echo
done

echo "Encoding sweep complete."
echo "Logs saved under ${LOG_DIR}"
