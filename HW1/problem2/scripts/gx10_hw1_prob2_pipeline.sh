#!/usr/bin/env bash
# HW1 Problem 2 — full training pipeline for gx10
#
# Run from the repo root on gx10, ideally inside a tmux session so you can
# detach and come back later:
#
#   tmux new -s prob2
#   bash HW1/problem2/scripts/gx10_hw1_prob2_pipeline.sh 2>&1 | tee prob2_pipeline.log
#   # Ctrl-b d  to detach;  tmux attach -t prob2  to reattach
#
# Override any default via env vars before calling the script, e.g.:
#   PROB2_EPOCHS=30 PROB2_EXPORT_EVERY=3 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults (override with env vars) ────────────────────────────────────────
TRACKING_URI="${PROB2_TRACKING_URI:-http://gx10-mlflow-server:5001}"
EPOCHS="${PROB2_EPOCHS:-50}"
SAMPLES="${PROB2_SAMPLES:-200}"
EXPORT_EVERY="${PROB2_EXPORT_EVERY:-5}"
RUNTIME_DIR="HW1/problem2/hf_space/runtime"
LOG_DIR="${RUNTIME_DIR}/logs"

# Each entry is "layers_explicit layers_reuploading"
CONFIGS=(
  "4 4"
  "6 6"
  "2 2"
  "4 8"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
hr() { echo "════════════════════════════════════════════════════════════════"; }

# Run one pipeline stage inside Docker, tee output to the run log.
# All containers join the MLflow network so every stage can reach the server.
run_stage() {
    local label="$1"; shift
    echo "[${RUN_NAME}] ── ${label}"
    GX10_DOCKER_NETWORK=gx10-mlflow \
    ./scripts/gx10_run_py.sh HW1/problem2/train.py "$@" 2>&1 | tee -a "${RUN_LOG}"
    echo "[${RUN_NAME}] ── ${label} ✓"
}

# ── Clean old runtime exports ─────────────────────────────────────────────────
hr
echo "[pipeline] Cleaning old runtime exports from ${RUNTIME_DIR}/"
rm -f  "${RUNTIME_DIR}"/*.json
rm -rf "${RUNTIME_DIR}"/chunks/
rm -rf "${RUNTIME_DIR}"/runs/
mkdir -p "${LOG_DIR}"
echo "[pipeline] Clean done."

# ── Start MLflow server ───────────────────────────────────────────────────────
hr
echo "[pipeline] Starting MLflow server"
./scripts/gx10_mlflow_server.sh start

# ── Run all configs ───────────────────────────────────────────────────────────
TOTAL=${#CONFIGS[@]}
IDX=0

for config in "${CONFIGS[@]}"; do
    read -r le lr <<< "${config}"
    IDX=$((IDX + 1))

    export RUN_NAME="q2-le${le}-lr${lr}-e${EPOCHS}"
    RUN_DIR="${RUNTIME_DIR}/runs/${RUN_NAME}"
    export RUN_LOG="${LOG_DIR}/${RUN_NAME}.log"

    hr
    echo "[pipeline] [${IDX}/${TOTAL}]  ${RUN_NAME}"
    echo "[pipeline] run-dir : ${RUN_DIR}"
    echo "[pipeline] log     : ${RUN_LOG}"
    hr

    mkdir -p "${RUN_DIR}"
    : > "${RUN_LOG}"   # truncate / create log file

    # 1 ── Prepare datasets (shared across methods for this run)
    run_stage "1/5 prepare" prepare \
        --run-dir "${RUN_DIR}" \
        --n-samples "${SAMPLES}"

    # 2 ── Train explicit quantum model
    run_stage "2/5 explicit (L=${le})" run \
        --run-dir  "${RUN_DIR}" \
        --method   explicit \
        --epochs   "${EPOCHS}" \
        --layers-explicit "${le}" \
        --viewer-export-every "${EXPORT_EVERY}" \
        --tracking-uri "${TRACKING_URI}" \
        --run-name "${RUN_NAME}"

    # 3 ── Train data-reuploading model
    run_stage "3/5 reuploading (L=${lr})" run \
        --run-dir  "${RUN_DIR}" \
        --method   reuploading \
        --epochs   "${EPOCHS}" \
        --layers-reuploading "${lr}" \
        --viewer-export-every "${EXPORT_EVERY}" \
        --tracking-uri "${TRACKING_URI}" \
        --run-name "${RUN_NAME}"

    # 4 ── Fit quantum kernel (no epochs — just one kernel matrix)
    run_stage "4/5 kernel" run \
        --run-dir  "${RUN_DIR}" \
        --method   kernel \
        --tracking-uri "${TRACKING_URI}" \
        --run-name "${RUN_NAME}"

    # 5 ── Assemble viewer JSON from all three method artifacts
    run_stage "5/5 assemble" assemble \
        --run-dir           "${RUN_DIR}" \
        --run-name          "${RUN_NAME}" \
        --viewer-export-path "${RUNTIME_DIR}"

    hr
    echo "[pipeline] [${IDX}/${TOTAL}] ${RUN_NAME} done"
    echo "[pipeline] viewer → ${RUNTIME_DIR}/${RUN_NAME}.json"
    echo "[pipeline] log    → ${RUN_LOG}"
done

hr
echo "[pipeline] All ${TOTAL} configs complete."
echo ""
echo "  Viewer exports : ${RUNTIME_DIR}/*.json"
echo "  Chunk files    : ${RUNTIME_DIR}/chunks/"
echo "  Stage logs     : ${LOG_DIR}/"
echo ""
echo "  Open viewer via SSH tunnel:"
echo "    ssh -L 8787:localhost:8787 gx10"
echo "    python -m http.server 8787 --directory HW1/problem2/hf_space"
hr
