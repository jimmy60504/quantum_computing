#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HW1_SWEEP_EPOCHS="${HW1_SWEEP_EPOCHS:-20}"
export HW1_SWEEP_QUBITS="${HW1_SWEEP_QUBITS:-2,3,4}"
export HW1_SWEEP_LAYERS="${HW1_SWEEP_LAYERS:-2,3,4}"

cd "${REPO_ROOT}"
./scripts/gx10_hw1_sweep_grid.sh
