# quantum_computing

This repository is set up for local Qiskit development with Conda.

## Environment

- Environment manager: Conda
- Environment name: `quantum-computing`
- Python version: `3.11`
- Main package: `qiskit==2.3.0`

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

## Why Conda instead of Docker

This project currently targets local development. Conda is the simpler choice
for Python version management, interactive notebooks, and package isolation.
Docker is only necessary later if you need a fully reproducible containerized
runtime or deployment target.

