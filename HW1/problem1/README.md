# HW1 Problem 1 — Data Reuploading Regression

Quantum regression on `f(x1, x2) = sin(exp(x1) + x2)` using data reuploading circuits. Three circuit families are compared:

| Variant | Description |
|---------|-------------|
| `quantum_exact` | Fixed 1-qubit construction that directly matches the target |
| `phase_learnable` | Same structure with a learnable phase shift |
| `scaled_exact` | Learnable scales and biases before the final phase shift |

**[Live viewer →](https://huggingface.co/spaces/jimmy60504/Data-Reuploading-Demo)**

## gx10 setup

Sync local code to `gx10` (run from repo root):

```bash
./scripts/sync_to_gx10.sh
```

Build the Docker image on `gx10`:

```bash
docker build -t quantum-gx10 .
```

The image base is `nvcr.io/nvidia/pytorch:26.03-py3`. GPU jobs use the wrapper script which mounts the repo at `/workspace`:

```bash
./scripts/gx10_run_py.sh <path/to/script.py> [args...]
```

## Training

Single training run:

```bash
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --encoding phase_learnable \
  --num-qubits 1 \
  --num-layers 1
```

Switch device and diff method:

```bash
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --device default.qubit \
  --diff-method backprop
```

## MLflow

Start the tracking server on `gx10`:

```bash
./scripts/gx10_mlflow_server.sh start
```

Point a training run at it:

```bash
GX10_DOCKER_NETWORK=gx10-mlflow \
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --tracking-uri http://gx10-mlflow-server:5001
```

Open the UI:

```bash
./scripts/gx10_mlflow_ui.sh
```

## Full pipeline

For the standard multi-run sweep (snapshot → evaluate → render → Fourier):

```bash
./HW1/problem1/scripts/gx10_hw1_prob1_full_pipeline.sh
```

Manual step-by-step:

```bash
# 1. Train in snapshot-only mode
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --render-mode snapshots-only \
  --viewer-export-every 1 \
  --encoding phase_learnable \
  --num-qubits 1 --num-layers 1 \
  --run-name phase-learnable-q1-l1-e20

# 2. Evaluate metrics
./scripts/gx10_run_py.sh HW1/problem1/tools/evaluate_snapshot_chunk.py \
  --snapshot-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json
./scripts/gx10_run_py.sh HW1/problem1/tools/merge_evaluated_chunks.py \
  --snapshot-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json \
  --require-complete

# 3. Render viewer output
./scripts/gx10_run_py.sh HW1/problem1/tools/render_snapshot_chunk.py \
  --snapshot-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json
./scripts/gx10_run_py.sh HW1/problem1/tools/merge_rendered_chunks.py \
  --snapshot-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json \
  --require-complete

# 4. Fourier analysis
./scripts/gx10_run_py.sh HW1/problem1/tools/fourier_analysis.py \
  --viewer-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20.json
```

See [`kb/distributed_render_workflow.md`](../../kb/distributed_render_workflow.md) for the snapshot-postprocess design.

## Upload to Hugging Face

```bash
./HW1/problem1/scripts/gx10_prepare_hf_space.sh
./HW1/problem1/scripts/gx10_upload_hf_space.sh jimmy60504/Data-Reuploading-Demo
```

Runtime data (chunks, logs) is kept in a separate dataset repo to stay under the 1 GB Space limit:

```bash
./HW1/problem1/scripts/gx10_upload_hf_runtime_dataset.sh jimmy60504/data-reuploading-demo-runtime
```
