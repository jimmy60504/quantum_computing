#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_ROOT="${REPO_ROOT}/logs/hw1_prob1_pipeline"
RUN_LOG_DIR="${LOG_ROOT}/runs"
mkdir -p "${RUN_LOG_DIR}"

EPOCHS="${HW1_PIPELINE_EPOCHS:-20}"
EXPERIMENT_NAME="${HW1_PIPELINE_EXPERIMENT_NAME:-hw1-problem1-datareuploading}"
TRACKING_URI="${HW1_PIPELINE_TRACKING_URI:-http://gx10-mlflow-server:5001}"
TRAIN_DEVICE="${HW1_PIPELINE_TRAIN_DEVICE:-default.qubit}"
TRAIN_DIFF_METHOD="${HW1_PIPELINE_TRAIN_DIFF_METHOD:-backprop}"
RENDER_DEVICE="${HW1_PIPELINE_RENDER_DEVICE:-lightning.qubit}"
RENDER_DIFF_METHOD="${HW1_PIPELINE_RENDER_DIFF_METHOD:-adjoint}"
VIEWER_EXPORT_EVERY="${HW1_PIPELINE_VIEWER_EXPORT_EVERY:-1}"
TRAIN_NUM_SAMPLES="${HW1_PIPELINE_NUM_SAMPLES:-1000}"
TRAIN_BATCH_SIZE="${HW1_PIPELINE_BATCH_SIZE:-64}"
TRAIN_HEATMAP_GRID_SIZE="${HW1_PIPELINE_HEATMAP_GRID_SIZE:-64}"
RENDER_WORKERS="${HW1_PIPELINE_RENDER_WORKERS:-20}"
RENDER_CPUSET="${HW1_PIPELINE_RENDER_CPUSET:-0-19}"
RENDER_CPUS="${HW1_PIPELINE_RENDER_CPUS:-20}"
TRAIN_CPUSET="${HW1_PIPELINE_TRAIN_CPUSET:-5-9,15-19}"
TRAIN_CPUS="${HW1_PIPELINE_TRAIN_CPUS:-10}"

RUN_MATRIX=(
  "raw 2 2"
  "raw 2 3"
  "raw 3 2"
  "raw 3 3"
  "poly 2 2"
  "exp 2 2"
)

run_case() {
  local encoding="$1"
  local qubits="$2"
  local layers="$3"
  local run_name="${encoding}-q${qubits}-l${layers}-e${EPOCHS}"
  local snapshot_export="hf_space_hw1_problem1/runtime/${run_name}_snapshots.json"
  local viewer_export="hf_space_hw1_problem1/runtime/${run_name}.json"
  local case_log="${RUN_LOG_DIR}/${run_name}.log"

  {
    echo "============================================================"
    echo "run_name=${run_name}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "============================================================"

    echo "[1/4] train snapshots"
    (
      cd "${REPO_ROOT}"
      GX10_DOCKER_NETWORK=gx10-mlflow \
      GX10_CPUSET="${TRAIN_CPUSET}" \
      GX10_CPUS="${TRAIN_CPUS}" \
      ./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
        --tracking-uri "${TRACKING_URI}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --run-name "${run_name}" \
        --encoding "${encoding}" \
        --num-qubits "${qubits}" \
        --num-layers "${layers}" \
        --epochs "${EPOCHS}" \
        --device "${TRAIN_DEVICE}" \
        --diff-method "${TRAIN_DIFF_METHOD}" \
        --render-mode snapshots-only \
        --viewer-export-every "${VIEWER_EXPORT_EVERY}" \
        --num-samples "${TRAIN_NUM_SAMPLES}" \
        --batch-size "${TRAIN_BATCH_SIZE}" \
        --heatmap-grid-size "${TRAIN_HEATMAP_GRID_SIZE}"
    )

    echo "[2/4] evaluate metrics"
    (
      cd "${REPO_ROOT}"
      PROB1_RENDER_WORKER_THREADS=1 \
      GX10_CPUSET="${RENDER_CPUSET}" \
      GX10_CPUS="${RENDER_CPUS}" \
      ./scripts/gx10_run_py.sh HW1/problem1/tools/evaluate_snapshot_chunk.py \
        --snapshot-export "${snapshot_export}" \
        --render-workers "${RENDER_WORKERS}" \
        --device "${RENDER_DEVICE}" \
        --diff-method "${RENDER_DIFF_METHOD}"
      GX10_CPUSET="${RENDER_CPUSET}" \
      GX10_CPUS="${RENDER_CPUS}" \
      ./scripts/gx10_run_py.sh HW1/problem1/tools/merge_evaluated_chunks.py \
        --snapshot-export "${snapshot_export}" \
        --require-complete
    )

    echo "[3/4] render viewer"
    (
      cd "${REPO_ROOT}"
      PROB1_RENDER_WORKER_THREADS=1 \
      GX10_CPUSET="${RENDER_CPUSET}" \
      GX10_CPUS="${RENDER_CPUS}" \
      ./scripts/gx10_run_py.sh HW1/problem1/tools/render_snapshot_chunk.py \
        --snapshot-export "${snapshot_export}" \
        --render-workers "${RENDER_WORKERS}" \
        --device "${RENDER_DEVICE}" \
        --diff-method "${RENDER_DIFF_METHOD}"
      GX10_CPUSET="${RENDER_CPUSET}" \
      GX10_CPUS="${RENDER_CPUS}" \
      ./scripts/gx10_run_py.sh HW1/problem1/tools/merge_rendered_chunks.py \
        --snapshot-export "${snapshot_export}" \
        --require-complete
    )

    echo "[4/4] fourier analysis"
    (
      cd "${REPO_ROOT}"
      GX10_CPUSET="${RENDER_CPUSET}" \
      GX10_CPUS="${RENDER_CPUS}" \
      ./scripts/gx10_run_py.sh HW1/problem1/tools/fourier_analysis.py \
        --viewer-export "${viewer_export}"
    )

    echo "completed_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "run_name=${run_name} done"
  } 2>&1 | tee "${case_log}"
}

main() {
  cd "${REPO_ROOT}"
  ./scripts/gx10_mlflow_server.sh start >/dev/null 2>&1 || ./scripts/gx10_mlflow_server.sh restart >/dev/null

  echo "HW1 Problem 1 full pipeline"
  echo "repo=${REPO_ROOT}"
  echo "tracking_uri=${TRACKING_URI}"
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "epochs=${EPOCHS}"
  echo "train_backend=${TRAIN_DEVICE}+${TRAIN_DIFF_METHOD}"
  echo "render_backend=${RENDER_DEVICE}+${RENDER_DIFF_METHOD}"
  echo "render_workers=${RENDER_WORKERS}"
  echo "log_root=${LOG_ROOT}"
  echo

  for spec in "${RUN_MATRIX[@]}"; do
    # shellcheck disable=SC2086
    run_case ${spec}
  done

  echo "all runs completed at $(date '+%Y-%m-%d %H:%M:%S %z')"
}

main "$@"
