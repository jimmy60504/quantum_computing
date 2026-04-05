# quantum_computing

This repository is set up for local Qiskit development with Conda and remote
GPU-oriented work with Docker on `gx10`.

## Environment

- Environment manager: Conda
- Environment name: `quantum-computing`
- Python version: `3.11`
- Main packages: `qiskit==2.3.0`, `pennylane==0.44.1`

## Docker on `gx10`

For the ARM64 NVIDIA `gx10` machine, prefer an `nvcr.io` base image instead of
a generic Docker Hub Python image.

The provided [Dockerfile](./Dockerfile) defaults to:

- `nvcr.io/nvidia/pytorch:26.03-py3`

As of `2026-04-02`, I verified on `gx10` that this `26.03` tag can be pulled
from `nvcr.io`.

Build:

```bash
docker build -t quantum-gx10 .
```

Build with a newer verified NVIDIA base:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3 \
  -t quantum-gx10 .
```

Run the smoke test:

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  quantum-gx10
```

Build the Qiskit Aer GPU image for `gx10`:

```bash
docker build -f Dockerfile.aer-gpu -t quantum-gx10:aer-gpu .
```

Run the Qiskit Aer GPU demo:

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  quantum-gx10:aer-gpu \
  python qiskit_aer_gpu_demo.py
```

Or run it on `gx10` through the wrapper:

```bash
ssh gx10
cd ~/quantum_computing
./scripts/gx10_aer_demo.sh
```

You can scale the demo circuit up when you want a heavier benchmark:

```bash
docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e AER_DEMO_QUBITS=24 \
  -e AER_DEMO_LAYERS=4 \
  quantum-gx10:aer-gpu \
  python qiskit_aer_gpu_demo.py
```

Wrapper scripts are also available for common `gx10` tasks:

```bash
./scripts/sync_to_gx10.sh
ssh gx10
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/sample.py
./scripts/gx10_run_py.sh pennylane_hello.py
GX10_IMAGE=quantum-gx10:aer-gpu ./scripts/gx10_run_py.sh qiskit_aer_gpu_demo.py
```

If you are working from your Mac and want to push the latest files to `gx10`:

```bash
./scripts/sync_to_gx10.sh
```

Then on `gx10`:

```bash
ssh gx10
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/sample.py
```

Open a shell in the container with the repo mounted:

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

The first image includes:

- `torch` from the NVIDIA base image
- `qiskit==2.3.0`
- `qiskit-machine-learning`
- `pennylane`
- the plotting and notebook-adjacent packages already used by this repository

Note:

- the local Conda workflow still targets Python `3.11`
- the current Docker image inherits Python `3.12` from NVIDIA's
  `nvcr.io/nvidia/pytorch:26.03-py3` base

## Qiskit Aer GPU note

On `gx10`, `qiskit-aer-gpu` is not available as a ready-made wheel because the
host is `aarch64`. The repository therefore includes
[Dockerfile.aer-gpu](./Dockerfile.aer-gpu), which builds `qiskit-aer` from
source with CUDA enabled and applies a small compatibility patch for the
current CUDA 13.2 toolchain on `gx10`. The Docker build also pins
`AER_CUDA_ARCH=8.6+PTX`, because GPU auto-detection is unavailable during
`docker build` and the default Aer fallback arch list is not compatible with
the CUDA 13.2 toolchain in the NVIDIA 26.03 base image.

## Create or update the environment

```bash
conda env create -f environment.yml
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

## Activate

```bash
conda activate quantum-computing
```

## Verify Qiskit

```bash
python -c "import qiskit; print(qiskit.__version__)"
```

## Verify PennyLane

```bash
python -c "import pennylane as qml; print(qml.__version__)"
```

## Run the local hello example

```bash
conda activate quantum-computing
python hello_qiskit.py
```

This example uses Qiskit's built-in `BasicSimulator`, so it runs locally on
your machine and does not send jobs to IBM Quantum.

## Run the PennyLane hello example

```bash
conda activate quantum-computing
python pennylane_hello.py
```

This example trains a 1-qubit circuit with gradient descent until its output is
close to the `|1>` state. It is meant to be the smallest useful PennyLane
example in the repository: device, QNode, differentiable parameter, loss
function, and optimizer are all present in one file.

## Run HW1 Problem 1 baseline

On `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py
```

This script now logs params, per-epoch MSE, and a loss-curve artifact to
MLflow. By default it uses a local SQLite backend at `./mlflow.db`.
The sync helper excludes `mlflow.db`, `mlartifacts/`, and `HW1/artifacts/`
because they are run outputs rather than source files.

You can switch PennyLane simulator and differentiation mode from the CLI:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --device default.qubit \
  --diff-method backprop
```

You can also compare encoding choices at fixed `q=2`, `l=2`:

```bash
cd ~/quantum_computing
./scripts/gx10_hw1_sweep_encodings_q2l2.sh
```

Or run one encoding explicitly:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --encoding exp \
  --num-qubits 2 \
  --num-layers 2
```

## Scaffold HW1 Problem 2

The repository now includes a first-pass scaffold under
[`HW1/problem2`](./HW1/problem2) for the three-method classification benchmark:

- explicit quantum model
- implicit quantum kernel method
- data reuploading circuit

It includes:

- dataset helpers for `circle` and `moons`
- shared benchmark/result interfaces
- a CLI that writes a dataset preview and a benchmark plan JSON

Run the scaffold preview locally:

```bash
conda activate quantum-computing
python -m HW1.problem2.scaffold --preview-datasets --write-plan
```

Or try the current faster-to-scale baseline:

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --device lightning.qubit \
  --diff-method adjoint
```

To inspect the runs on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_mlflow_ui.sh
```

## Hugging Face static viewer

The repository includes a static Hugging Face Space export scaffold in
[hf_space_hw1_problem1](/Users/jimmy/Library/CloudStorage/OneDrive-Personal/Code/quantum_computing/hf_space_hw1_problem1).

Recommended workflow:

- Sync code from your Mac to `gx10`
- Run training on `gx10`
- Keep generated viewer data on `gx10` under `hf_space_hw1_problem1/runtime/`
- The viewer export is batch-based: one slider step per exported batch
- The viewer also keeps a runtime manifest so you can switch between multiple
  hyperparameter runs from a dropdown
- Only prepare a publish bundle when you are ready to upload to Hugging Face

Preview it on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_hf_viewer.sh
```

Prepare a publish bundle on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_prepare_hf_space.sh
```

Preview it locally if needed:

```bash
cd hf_space_hw1_problem1
python3 -m http.server 8000
```

If you want a different port:

```bash
cd ~/quantum_computing
MLFLOW_PORT=5001 ./scripts/gx10_mlflow_ui.sh
```

## Run the QFT demo

```bash
conda activate quantum-computing
python qft_demo.py
```

This also runs locally with Qiskit's built-in simulator.

## 研究知識庫（KB）

專案已在 `kb/` 提供一套輕量知識庫，先用於研究方向整理：

- `kb/README.md`：KB 使用方式
- `kb/research_directions.md`：研究方向清單
- `kb/directions/`：方向詳細筆記
- `kb/experiment_plans.md`：實驗規劃（目前暫不啟用）
- `kb/templates/`：可重用模板

入口：

```bash
open kb/README.md
```

## Why Conda instead of Docker

Conda remains the default for local macOS development. Docker is now the
preferred path for reproducible work on the remote `gx10` NVIDIA machine.
