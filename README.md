# quantum_computing

This repository is set up for local Qiskit development with Conda and remote
GPU-oriented work with Docker on `gx10`.

## Environment

- Environment manager: Conda
- Environment name: `quantum-computing`
- Python version: `3.11`
- Main package: `qiskit==2.3.0`

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

## Run the local hello example

```bash
conda activate quantum-computing
python hello_qiskit.py
```

This example uses Qiskit's built-in `BasicSimulator`, so it runs locally on
your machine and does not send jobs to IBM Quantum.

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
