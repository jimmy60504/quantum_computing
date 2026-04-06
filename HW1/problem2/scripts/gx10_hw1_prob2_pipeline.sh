#!/usr/bin/env bash
# HW1 Problem 2 — full training pipeline for gx10 (parallel methods)
#
# Run from the repo root on gx10 inside a tmux session:
#
#   tmux new -s prob2
#   bash HW1/problem2/scripts/gx10_hw1_prob2_pipeline.sh 2>&1 | tee prob2_pipeline.log
#   # Ctrl-b d  to detach;  tmux attach -t prob2  to reattach
#
# For each config the three QML methods (explicit / reuploading / kernel)
# are launched simultaneously after prepare, then assemble waits for all
# three to finish.  This cuts wall-clock time to ~1/3 vs sequential.
#
# Override defaults with env vars:
#   PROB2_EPOCHS=30 PROB2_EXPORT_EVERY=3 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TRACKING_URI="${PROB2_TRACKING_URI:-http://gx10-mlflow-server:5001}"
EPOCHS="${PROB2_EPOCHS:-50}"
SAMPLES="${PROB2_SAMPLES:-200}"
EXPORT_EVERY="${PROB2_EXPORT_EVERY:-5}"
RUNTIME_DIR="HW1/problem2/hf_space/runtime"
LOG_DIR="${RUNTIME_DIR}/logs"

# "layers_explicit layers_reuploading"
CONFIGS=(
  "4 4"
  "6 6"
  "2 2"
  "4 8"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
hr() { echo "════════════════════════════════════════════════════════════════"; }
ts() { date "+%H:%M:%S"; }

# Synchronous stage — output tees to both stdout and the run log.
run_stage() {
    local label="$1"; shift
    echo "[$(ts)] [${RUN_NAME}] ── ${label}"
    GX10_DOCKER_NETWORK=gx10-mlflow \
    ./scripts/gx10_run_py.sh HW1/problem2/train.py "$@" 2>&1 | tee -a "${RUN_LOG}"
    echo "[$(ts)] [${RUN_NAME}] ── ${label} ✓"
}

# Background method stage — output goes to a dedicated per-method log file.
# Prints the PID so the caller can wait on it.
run_method_bg() {
    local method="$1"; shift
    local log="${LOG_DIR}/${RUN_NAME}-${method}.log"
    : > "${log}"
    echo "[$(ts)] [${RUN_NAME}] ── ${method} → background (log: ${log})"
    GX10_DOCKER_NETWORK=gx10-mlflow \
    ./scripts/gx10_run_py.sh HW1/problem2/train.py run \
        --run-dir          "${RUN_DIR}" \
        --method           "${method}" \
        --tracking-uri     "${TRACKING_URI}" \
        --run-name         "${RUN_NAME}" \
        "$@" \
        > "${log}" 2>&1 &
    echo $!
}

# ── Clean old runtime exports ─────────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Cleaning old runtime exports from ${RUNTIME_DIR}/"
rm -f  "${RUNTIME_DIR}"/*.json
rm -rf "${RUNTIME_DIR}"/chunks/
rm -rf "${RUNTIME_DIR}"/runs/
mkdir -p "${LOG_DIR}"
echo "[$(ts)] [pipeline] Clean done."

# ── Start MLflow server ───────────────────────────────────────────────────────
hr
echo "[$(ts)] [pipeline] Starting MLflow server"
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
    echo "[$(ts)] [pipeline] [${IDX}/${TOTAL}]  ${RUN_NAME}"
    hr

    mkdir -p "${RUN_DIR}"
    : > "${RUN_LOG}"

    # ── Stage 1: prepare (sequential — methods depend on datasets.npz) ─────────
    run_stage "1/3 prepare" prepare \
        --run-dir   "${RUN_DIR}" \
        --n-samples "${SAMPLES}"

    # ── Stage 2: train all three methods in parallel ───────────────────────────
    echo "[$(ts)] [${RUN_NAME}] ── 2/3 launching methods in parallel..."

    PID_E=$(run_method_bg explicit \
        --epochs              "${EPOCHS}" \
        --layers-explicit     "${le}" \
        --viewer-export-every "${EXPORT_EVERY}")

    PID_R=$(run_method_bg reuploading \
        --epochs               "${EPOCHS}" \
        --layers-reuploading   "${lr}" \
        --viewer-export-every  "${EXPORT_EVERY}")

    PID_K=$(run_method_bg kernel)   # no epochs — one kernel matrix fit

    echo "[$(ts)] [${RUN_NAME}]    explicit    PID=${PID_E}"
    echo "[$(ts)] [${RUN_NAME}]    reuploading PID=${PID_R}"
    echo "[$(ts)] [${RUN_NAME}]    kernel      PID=${PID_K}"

    # Wait for all three; collect exit codes individually so we can report
    # which method failed rather than just aborting on the first one.
    FAIL=0
    wait "${PID_E}" || { echo "[$(ts)] [${RUN_NAME}] ERROR: explicit failed"; FAIL=1; }
    wait "${PID_R}" || { echo "[$(ts)] [${RUN_NAME}] ERROR: reuploading failed"; FAIL=1; }
    wait "${PID_K}" || { echo "[$(ts)] [${RUN_NAME}] ERROR: kernel failed"; FAIL=1; }

    if [[ "${FAIL}" -ne 0 ]]; then
        echo "[$(ts)] [${RUN_NAME}] One or more methods failed — skipping assemble."
        echo "  Check per-method logs in ${LOG_DIR}/${RUN_NAME}-*.log"
        continue   # skip to next config instead of aborting the whole pipeline
    fi

    echo "[$(ts)] [${RUN_NAME}] ── 2/3 all methods done ✓"

    # ── Stage 3: assemble viewer JSON (sequential — needs all three artifacts) ─
    run_stage "3/3 assemble" assemble \
        --run-dir            "${RUN_DIR}" \
        --run-name           "${RUN_NAME}" \
        --viewer-export-path "${RUNTIME_DIR}"

    hr
    echo "[$(ts)] [pipeline] [${IDX}/${TOTAL}] ${RUN_NAME} done"
    echo "  viewer  → ${RUNTIME_DIR}/${RUN_NAME}.json"
    echo "  chunks  → ${RUNTIME_DIR}/chunks/${RUN_NAME}_epoch_*.json"
    echo "  logs    → ${LOG_DIR}/${RUN_NAME}*.log"
done

hr
echo "[$(ts)] [pipeline] All ${TOTAL} configs complete."
echo ""
echo "  Viewer exports : ${RUNTIME_DIR}/*.json"
echo "  Stage logs     : ${LOG_DIR}/"
echo ""
echo "  Open viewer via SSH tunnel:"
echo "    ssh -L 8787:localhost:8787 gx10"
echo "    python -m http.server 8787 --directory HW1/problem2/hf_space"
hr
