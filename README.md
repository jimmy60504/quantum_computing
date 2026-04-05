# quantum_computing

Local Qiskit and PennyLane experiments with a Conda workflow on macOS and a
Docker-based execution workflow on `gx10`.

## At a glance

- Local source of truth: this repository on your Mac
- Local environment manager: Conda
- Conda environment: `quantum-computing`
- Python: `3.11`
- Core packages: `qiskit==2.3.0`, `pennylane==0.44.1`, `torch`
- Remote experiment host: `gx10`
- Remote execution model: Docker, with the repo synced from local source

The intended workflow is:

1. Edit code locally.
2. Sync to `gx10` when you want to run heavier experiments.
3. Treat `gx10` as an execution mirror for runs and generated artifacts.
4. Keep code changes and commits anchored in the local repo.

## Repository layout

- `HW1/problem1/`: current regression experiments, training pipeline, render tools
- `HW1/problem2/`: classification scaffold
- `hf_space_hw1_problem1/`: static Hugging Face viewer scaffold
- `scripts/`: sync, Docker-run wrappers, MLflow helpers, viewer helpers
- `kb/`: lightweight research notes and templates
- `hello_qiskit.py`, `pennylane_hello.py`, `qft_demo.py`: small local demos

## Local setup

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Update an existing environment:

```bash
conda env update -f environment.yml --prune
```

Activate it:

```bash
conda activate quantum-computing
```

Quick sanity checks:

```bash
python -c "import qiskit; print(qiskit.__version__)"
python -c "import pennylane as qml; print(qml.__version__)"
python -c "import torch; print(torch.__version__)"
```

## Local demos

Qiskit hello world:

```bash
conda activate quantum-computing
python hello_qiskit.py
```

PennyLane hello world:

```bash
conda activate quantum-computing
python pennylane_hello.py
```

QFT demo:

```bash
conda activate quantum-computing
python qft_demo.py
```

## `gx10` workflow

Sync local code to the remote execution mirror:

```bash
./scripts/sync_to_gx10.sh
```

Then connect:

```bash
ssh gx10
cd ~/quantum_computing
```

The sync helper intentionally excludes runtime-heavy outputs such as:

- `mlruns`
- `mlartifacts`
- `mlflow.db`
- `HW1/artifacts`
- `hf_space_hw1_problem1/runtime`

That keeps local source and remote experiment artifacts separate.

## Docker on `gx10`

For `gx10`, prefer the provided NVIDIA ARM64 Docker workflow over bare-host
Python.

Build the main image:

```bash
docker build -t quantum-gx10 .
```

The default base image is:

- `nvcr.io/nvidia/pytorch:26.03-py3`

Run a Python file inside the repo-mounted container through the wrapper:

```bash
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py
```

Useful wrapper notes:

- heavy jobs default to CPU set `5-9,15-19` with `10` CPUs
- you can override with `GX10_CPUSET` and `GX10_CPUS`
- set `GX10_DOCKER_NETWORK` when the job needs to reach containerized services
  such as MLflow

Open a shell in the container if needed:

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  quantum-gx10 \
  bash
```

## Qiskit Aer GPU on `gx10`

`qiskit-aer-gpu` is not available as a ready-made wheel on `gx10` because the
host is `aarch64`. This repo therefore includes
[`Dockerfile.aer-gpu`](./Dockerfile.aer-gpu), which builds Aer with CUDA
support for that environment.

Build it:

```bash
docker build -f Dockerfile.aer-gpu -t quantum-gx10:aer-gpu .
```

Run the demo:

```bash
./scripts/gx10_aer_demo.sh
```

Or directly:

```bash
GX10_IMAGE=quantum-gx10:aer-gpu ./scripts/gx10_run_py.sh qiskit_aer_gpu_demo.py
```

## HW1 Problem 1

The main active experiment lives in `HW1/problem1/`. It is a data reuploading
regression setup with multiple classical encodings including `raw`, `poly`, and
`exp`.

### One-off training run

Run a single training job on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py
```

Run a specific encoding:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --encoding exp \
  --num-qubits 2 \
  --num-layers 2
```

You can also switch backend and differentiation mode:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --device default.qubit \
  --diff-method backprop
```

### MLflow

Start the remote MLflow service on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_mlflow_server.sh start
```

Point training at that tracking server:

```bash
cd ~/quantum_computing
GX10_DOCKER_NETWORK=gx10-mlflow \
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --tracking-uri http://gx10-mlflow-server:5001
```

Open the UI:

```bash
cd ~/quantum_computing
./scripts/gx10_mlflow_ui.sh
```

Check server status or logs:

```bash
cd ~/quantum_computing
./scripts/gx10_mlflow_server.sh status
./scripts/gx10_mlflow_server.sh logs
```

### Snapshot-first workflow

For longer runs, the default pattern is:

1. Train and save snapshots.
2. Evaluate per-step metrics from snapshots.
3. Render viewer exports from snapshots.
4. Run Fourier analysis on the final viewer export.

Train in snapshot-only mode:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --render-mode snapshots-only \
  --viewer-export-every 1 \
  --run-name raw-q2-l2-e20
```

Evaluate and merge metrics:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/tools/evaluate_snapshot_chunk.py \
  --snapshot-export hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json

./scripts/gx10_run_py.sh HW1/problem1/tools/merge_evaluated_chunks.py \
  --snapshot-export hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json \
  --require-complete
```

Render and merge viewer output:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/tools/render_snapshot_chunk.py \
  --snapshot-export hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json

./scripts/gx10_run_py.sh HW1/problem1/tools/merge_rendered_chunks.py \
  --snapshot-export hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json \
  --require-complete
```

Run Fourier analysis:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/tools/fourier_analysis.py \
  --viewer-export hf_space_hw1_problem1/runtime/raw-q2-l2-e20.json
```

### Full pipeline helper

For the standard multi-run workflow, use:

```bash
cd ~/quantum_computing
./scripts/gx10_hw1_prob1_full_pipeline.sh
```

This runs the configured matrix in the script and produces:

- snapshot exports
- merged metrics exports
- viewer exports
- Fourier analysis outputs

See [`kb/distributed_render_workflow.md`](./kb/distributed_render_workflow.md)
for more detail on the snapshot-postprocess design.

## HW1 Problem 2

`HW1/problem2/` is a scaffold for a three-method classification benchmark:

- explicit quantum model
- implicit quantum kernel method
- data reuploading circuit

Preview datasets and write the plan:

```bash
conda activate quantum-computing
python -m HW1.problem2.scaffold --preview-datasets --write-plan
```

## Hugging Face viewer

The static viewer scaffold lives in
[`hf_space_hw1_problem1/`](./hf_space_hw1_problem1).

Recommended flow:

1. Sync code to `gx10`.
2. Run training and post-processing on `gx10`.
3. Keep generated runtime files under `hf_space_hw1_problem1/runtime/` on `gx10`.
4. Only prepare a publish bundle when you want to upload to Hugging Face.

Preview on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_hf_viewer.sh
```

Prepare a publish bundle:

```bash
cd ~/quantum_computing
./scripts/gx10_prepare_hf_space.sh
```

Preview the static scaffold locally if needed:

```bash
cd hf_space_hw1_problem1
python3 -m http.server 8000
```

## Knowledge base

The repository includes a lightweight project KB under `kb/`:

- `kb/README.md`: KB usage
- `kb/research_directions.md`: research direction index
- `kb/directions/`: detailed direction notes
- `kb/experiment_plans.md`: experiment planning notes
- `kb/templates/`: reusable templates

Open the KB entry point:

```bash
open kb/README.md
```

## Why Conda locally and Docker remotely

Conda remains the default for local macOS development. Docker is the preferred
path for reproducible execution on the remote ARM64 NVIDIA `gx10` machine.
