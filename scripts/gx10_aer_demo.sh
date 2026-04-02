#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GX10_IMAGE=quantum-gx10:aer-gpu "${SCRIPT_DIR}/gx10_run_py.sh" qiskit_aer_gpu_demo.py "$@"
