# Compute Hosts

This note tracks remote machines that are relevant to this repository.

## `gx10`

- Last verified: `2026-03-25`
- SSH entry: `ssh gx10`
- Remote user/host observed after login: `jimmy@gx10-dab6`
- Purpose: remote GPU-capable machine for heavier experiments and setup work

### Observed environment

- OS: Ubuntu `24.04.4 LTS`
- Kernel: `6.17.0-1014-nvidia`
- Architecture: `aarch64`
- CPU: 20 cores
- Memory: `121 GiB`
- Root disk: about `916G` total, about `717G` available at verification time
- GPU: `NVIDIA GB10`
- NVIDIA driver: `580.142`
- CUDA runtime reported by `nvidia-smi`: `13.0`
- `nvcc`: available (`13.0.88`)

### Python and tooling status

- System Python: `3.12.3` at `/usr/bin/python3`
- `pip3`: available
- `git`: available
- `docker`: available
- NVIDIA container workflow is preferred over a plain Docker Hub Python base on
  this host
- `conda`: not installed at verification time
- `mamba`: not installed at verification time

### Container guidance for this host

- Treat this machine as an NVIDIA/DGX-style environment.
- Prefer NVIDIA Container Registry images from `nvcr.io` as the Docker base when
  containerizing workloads for `gx10`.
- Because this host is `aarch64`, avoid assuming a generic `docker.io` Python
  base image is the best fit for GPU work.
- For CUDA-enabled containers, prefer an NVIDIA CUDA Ubuntu 24.04 image that
  supports ARM64 and matches the host CUDA/driver stack closely enough for the
  workload.
- If repo code is being prepared specifically for `gx10`, document the chosen
  `nvcr.io` base image and the intended `docker run --gpus all ...` entry path.

### Repo-specific note

This repository targets Conda with:

- environment name: `quantum-computing`
- Python: `3.11`
- dependencies from `environment.yml`

For local development on macOS, Conda remains the default path. For `gx10`,
containerization is now the preferred path when running repository code on the
remote machine.
