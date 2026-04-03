#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/hw1_sweeps"

DEVICE="${HW1_SWEEP_DEVICE:-default.qubit}"
DIFF_METHOD="${HW1_SWEEP_DIFF_METHOD:-backprop}"
EPOCHS="${HW1_SWEEP_EPOCHS:-10}"
BATCH_SIZE="${HW1_SWEEP_BATCH_SIZE:-64}"
NUM_SAMPLES="${HW1_SWEEP_NUM_SAMPLES:-1000}"
HEATMAP_GRID_SIZE="${HW1_SWEEP_HEATMAP_GRID_SIZE:-24}"
RENDER_WORKERS="${HW1_SWEEP_RENDER_WORKERS:-16}"
VIEWER_EXPORT_EVERY="${HW1_SWEEP_VIEWER_EXPORT_EVERY:-1}"
QUBITS_CSV="${HW1_SWEEP_QUBITS:-2,4,8}"
LAYERS_CSV="${HW1_SWEEP_LAYERS:-2,4,8}"

IFS=',' read -r -a QUBITS <<< "${QUBITS_CSV}"
IFS=',' read -r -a LAYERS <<< "${LAYERS_CSV}"

mkdir -p "${LOG_DIR}"

echo "HW1 grid sweep starting"
echo "repo=${REPO_ROOT}"
echo "device=${DEVICE} diff_method=${DIFF_METHOD}"
echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_samples=${NUM_SAMPLES}"
echo "heatmap_grid_size=${HEATMAP_GRID_SIZE} render_workers=${RENDER_WORKERS}"
echo "qubits=${QUBITS_CSV} layers=${LAYERS_CSV}"
echo "logs=${LOG_DIR}"
echo

for q in "${QUBITS[@]}"; do
  for l in "${LAYERS[@]}"; do
    RUN_NAME="q${q}-l${l}-e${EPOCHS}"
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
        --num-qubits "${q}" \
        --num-layers "${l}" \
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
done

echo "Grid sweep complete."
echo "Logs saved under ${LOG_DIR}"
