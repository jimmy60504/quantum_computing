# quantum_computing

Qiskit and PennyLane experiments in quantum machine learning. Built for the QCAA course (HW1). Experiments run locally via Conda and remotely on `gx10` inside Docker.

## Live demos

| Problem | Space | What it shows |
|---------|-------|---------------|
| HW1 Problem 1 | [Data Reuploading Explorer](https://huggingface.co/spaces/jimmy60504/Data-Reuploading-Demo) | Quantum regression with data reuploading circuits |
| HW1 Problem 2 | [QML Classifier Explorer](https://huggingface.co/spaces/jimmy60504/QML-Classifier-Explorer) | Explicit / kernel / reuploading classifiers compared |
| HW1 Problem 3 | [Hybrid QNN Explorer](https://huggingface.co/spaces/jimmy60504/Hybrid-QNN-Explorer) | Hybrid QNN vs MLP on CIFAR-10 |

## Repository layout

| Path | Description |
|------|-------------|
| [`HW1/problem1/`](HW1/problem1/) | Quantum regression — data reuploading circuits |
| [`HW1/problem2/`](HW1/problem2/) | QML classifiers — explicit / kernel / reuploading |
| [`HW1/problem3/`](HW1/problem3/) | Hybrid CNN+QNN vs CNN+MLP on CIFAR-10 |
| `scripts/` | gx10 sync, Docker-run wrappers, MLflow helpers |
| `docker/` | Dependency files for the gx10 Docker image |
| `kb/` | Research notes and experiment plans |

## Environment

```bash
conda env create -f environment.yml
conda activate quantum-computing
```

See each problem's README for how to run training and upload results.
