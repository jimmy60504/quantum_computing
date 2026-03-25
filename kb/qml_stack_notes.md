# QML Stack Notes

This note captures the current working conclusion for the repository's quantum
machine learning stack and future container strategy.

## Current conclusion

- Learn fundamentals with `Qiskit`
- Build quantum machine learning workflows mainly with `PennyLane`
- Revisit `CUDA-Q` later if GPU-accelerated simulation or hybrid execution
  becomes important on `gx10`

## Why this split

- `Qiskit` is a good entry point for circuits, gates, transpilation, and core
  quantum computing concepts.
- `PennyLane` is a better candidate for the main QML workflow because it fits
  hybrid ML patterns more naturally.
- `CUDA-Q` looks promising for performance-oriented work on NVIDIA hardware, but
  it does not need to be the first framework introduced into this repository.

## Container strategy for a future image

- Build a Docker image for remote `gx10` work instead of relying on bare-host
  Python setup.
- Prefer an NVIDIA Container Registry (`nvcr.io`) base image instead of a
  generic Docker Hub Python image.
- Because `gx10` is an ARM64 NVIDIA machine, choose a base image that is known
  to support `aarch64` and aligns with the host CUDA stack.
- For a QML-oriented image, start from an NVIDIA PyTorch or CUDA base and then
  add the Python quantum stack.

## First image scope

The first image does not need to include everything possible. The preferred
first scope is:

- Python `3.11`
- `PennyLane`
- `Qiskit`
- `qiskit-machine-learning`
- `PyTorch`
- Jupyter support if notebook work becomes part of the workflow

`CUDA-Q` should remain optional for the first image unless there is a clear need
to benchmark or accelerate larger simulation workloads on `gx10`.
