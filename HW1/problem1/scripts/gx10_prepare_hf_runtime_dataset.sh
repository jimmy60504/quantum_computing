#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/HW1/problem1/hf_space/runtime"
OUTPUT_DIR="${HF_RUNTIME_DATASET_OUTPUT_DIR:-${REPO_ROOT}/.out/hw1_problem1_hf_runtime_dataset}"
MANIFEST_PATH="${RUNTIME_DIR}/viewer_manifest.json"
INCLUDE_RUN_IDS="${HF_RUNTIME_DATASET_INCLUDE_RUN_IDS:-}"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/runtime/chunks"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing runtime manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

python3 - "${MANIFEST_PATH}" "${OUTPUT_DIR}/runtime/viewer_manifest.json" "${INCLUDE_RUN_IDS}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
include_run_ids = [item.strip() for item in sys.argv[3].split(",") if item.strip()]

manifest = json.loads(manifest_path.read_text())
runs = manifest.get("runs", [])

if include_run_ids:
    include_set = set(include_run_ids)
    runs = [run for run in runs if run.get("id") in include_set]

if not runs:
    raise SystemExit("No runs selected for HF runtime dataset bundle.")

default_run = manifest.get("default_run")
if default_run not in {run.get("id") for run in runs}:
    default_run = runs[0].get("id")

filtered_manifest = {
    "title": manifest.get("title", "Data Reuploading Runs"),
    "default_run": default_run,
    "runs": runs,
}
output_path.write_text(json.dumps(filtered_manifest, indent=2))
PY

mapfile -t SELECTED_RUN_IDS < <(
  python3 - "${OUTPUT_DIR}/runtime/viewer_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for run in manifest.get("runs", []):
    run_id = run.get("id")
    if run_id:
        print(run_id)
PY
)

for run_id in "${SELECTED_RUN_IDS[@]}"; do
  [[ -z "${run_id}" ]] && continue

  for asset in \
    "${run_id}.json" \
    "${run_id}_circuit.png" \
    "${run_id}_fourier_spectrum.png"; do
    if [[ -f "${RUNTIME_DIR}/${asset}" ]]; then
      cp "${RUNTIME_DIR}/${asset}" "${OUTPUT_DIR}/runtime/${asset}"
    fi
  done

  while IFS= read -r chunk_path; do
    [[ -z "${chunk_path}" ]] && continue
    cp "${chunk_path}" "${OUTPUT_DIR}/runtime/chunks/"
  done < <(find "${RUNTIME_DIR}/chunks" -maxdepth 1 -type f -name "${run_id}_epoch_*.json" | sort)
done

cat > "${OUTPUT_DIR}/README.md" <<'EOF'
---
license: mit
task_categories:
- other
pretty_name: Data Reuploading Demo Runtime
---

# Data Reuploading Demo Runtime

This dataset repo stores runtime artifacts for the static Space viewer.

- `runtime/viewer_manifest.json`: manifest consumed by the viewer
- `runtime/*.json`: run summaries
- `runtime/chunks/`: per-epoch viewer chunks
- `runtime/*_circuit.png`: circuit previews
- `runtime/*_fourier_spectrum.png`: Fourier plots
- Snapshot exports, metrics chunks, and other build-only artifacts are omitted.
EOF

echo "Prepared HF runtime dataset bundle at: ${OUTPUT_DIR}"
