# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# The NVIDIA PyTorch image ships with a pip constraints file. We only drop the
# hard pin on packaging tools so adding the repo's Python stack stays practical.
RUN python - <<'PY'
from pathlib import Path

constraint_path = Path("/etc/pip/constraint.txt")
if constraint_path.exists():
    lines = []
    for line in constraint_path.read_text().splitlines():
        if line.startswith(("pip==", "setuptools==", "wheel==")):
            continue
        lines.append(line)
    constraint_path.write_text("\n".join(lines) + "\n")
PY

COPY docker/requirements-gx10.txt /tmp/requirements-gx10.txt

RUN python -m pip install -r /tmp/requirements-gx10.txt

COPY hello_qiskit.py qft_demo.py smoke_test.py README.md ./

CMD ["python", "smoke_test.py"]
