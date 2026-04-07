#!/usr/bin/env bash
# HW1 Problem 3 — full training pipeline for gx10 (MLP baseline + QNN hybrid)
#
# Run from the repo root on gx10 inside a tmux session:
#
#   tmux new -s prob3
#   bash HW1/problem3/scripts/gx10_hw1_prob3_pipeline.sh 2>&1 | tee prob3_pipeline.log
#   # Ctrl-b d  to detach;  tmux attach -t prob3  to reattach
#
# The MLP baseline and QNN hybrid are trained sequentially (QNN is slow on
# default.qubit; parallelising would not help much). After both complete,
# assemble merges their artifacts into a single viewer JSON.
#
# Override defaults with env vars:
#   PROB3_EPOCHS=10 PROB3_QUBITS=4 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TRACKING_URI="${PROB3_TRACKING_URI:-http://gx10-mlflow-server:5001}"
EPOCHS="${PROB3_EPOCHS:-20}"
BATCH_SIZE="${PROB3_BATCH_SIZE:-64}"
LR="${PROB3_LR:-1e-3}"
QUBITS="${PROB3_QUBITS:-8}"
LAYERS="${PROB3_LAYERS:-4}"
FEATURE_DIM="${PROB3_FEATURE_DIM:-256}"
DEVICE="${PROB3_DEVICE:-cpu}"
EXPORT_EVERY="${PROB3_EXPORT_EVERY:-1}"
RUNTIME_DIR="HW1/problem3/hf_space/runtime"
LOG_DIR="${RUNTIME_DIR}/logs"

RUN_NAME="${PROB3_RUN_NAME:-q${QUBITS}-l${LAYERS}-e${EPOCHS}}"

# ── Helpers ───────────────────────────────────────────────────────────────────
hr() { echo "════════════════════════════════════════════════════════════════"; }
ts() { date "+%H:%M:%S"; }

run_stage() {
    local label="$1"; shift
    echo "[$(ts)] [${RUN_NAME}] ── ${label}"
    GX10_DOCKER_NETWORK=gx10-mlflow \
    ./scripts/gx10_run_py.sh HW1/problem3/train.py "$@" 2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}.log"
    echo "[$(ts)] [${RUN_NAME}] ── ${label} ✓"
}

# ── Clean old runtime exports ─────────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Cleaning old runtime exports from ${RUNTIME_DIR}/"
rm -f  "${RUNTIME_DIR}"/*.json
rm -rf "${RUNTIME_DIR}"/runs/
mkdir -p "${LOG_DIR}"
echo "[$(ts)] [pipeline] Clean done."

# ── Start MLflow server ───────────────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Starting MLflow server"
./scripts/gx10_mlflow_server.sh start

# ── Common args ───────────────────────────────────────────────────────────────
RUN_DIR="${RUNTIME_DIR}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
: > "${LOG_DIR}/${RUN_NAME}.log"

COMMON_ARGS=(
    --epochs              "${EPOCHS}"
    --batch-size          "${BATCH_SIZE}"
    --learning-rate       "${LR}"
    --num-qubits          "${QUBITS}"
    --num-layers          "${LAYERS}"
    --feature-dim         "${FEATURE_DIM}"
    --device              "${DEVICE}"
    --viewer-export-every "${EXPORT_EVERY}"
    --viewer-export-path  "${RUNTIME_DIR}"
    --tracking-uri        "${TRACKING_URI}"
    --run-name            "${RUN_NAME}"
)

# ── Stage 1: Train MLP baseline ──────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Stage 1/3: Training MLP baseline"
run_stage "1/3 mlp" run --run-dir "${RUN_DIR}" --method mlp "${COMMON_ARGS[@]}"

# ── Stage 2: Train QNN hybrid ────────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Stage 2/3: Training QNN hybrid"
run_stage "2/3 qnn" run --run-dir "${RUN_DIR}" --method qnn "${COMMON_ARGS[@]}"

# ── Stage 3: Assemble viewer JSON ────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Stage 3/3: Assembling viewer"
run_stage "3/3 assemble" assemble --run-dir "${RUN_DIR}" \
    --viewer-export-path "${RUNTIME_DIR}" \
    --run-name "${RUN_NAME}" \
    --tracking-uri "${TRACKING_URI}"

hr
echo "[$(ts)] [pipeline] ${RUN_NAME} complete."
echo ""
echo "  Viewer export : ${RUNTIME_DIR}/${RUN_NAME}.json"
echo "  Stage logs    : ${LOG_DIR}/"
echo ""
echo "  Open viewer via SSH tunnel:"
echo "    ssh -L 8787:localhost:8787 gx10"
echo "    python -m http.server 8787 --directory HW1/problem3/hf_space"
hr
