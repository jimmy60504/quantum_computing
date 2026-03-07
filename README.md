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

## Why Conda instead of Docker

This project currently targets local development. Conda is the simpler choice
for Python version management, interactive notebooks, and package isolation.
Docker is only necessary later if you need a fully reproducible containerized
runtime or deployment target.
