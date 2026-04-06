"""Configuration helpers for HW1 Problem 3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HF_SPACE_DIR = Path("HW1") / "problem3" / "hf_space"
RUNTIME_DIR = HF_SPACE_DIR / "runtime"


@dataclass
class Prob3Config:
    seed: int = 11224001
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 64

    # CNN backbone
    backbone: str = "simple"  # "simple" or "resnet18"
    feature_dim: int = 256
    freeze_backbone: bool = False

    # Quantum circuit
    num_qubits: int = 8
    num_layers: int = 4
    q_device_name: str = "default.qubit"
    q_diff_method: str = "backprop"

    # Viewer export
    viewer_export_every: int = 1  # export every N epochs
    viewer_export_path: str = str(RUNTIME_DIR)
    confusion_grid_size: int = 10  # CIFAR-10 classes

    # MLflow
    tracking_uri: str = ""
    experiment_name: str = "hw1-problem3-hybrid-qnn"
    run_name: str = ""

    # Device
    device: str = "cpu"  # "cpu" or "cuda"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "prob3-run"


def make_default_export_stem(config: Prob3Config) -> str:
    if config.run_name:
        return slugify(config.run_name)
    return slugify(
        f"q{config.num_qubits}-l{config.num_layers}-e{config.epochs}"
    )


def resolve_viewer_export_path(config: Prob3Config) -> Path:
    root = Path(config.viewer_export_path)
    if root.suffix == ".json":
        root.parent.mkdir(parents=True, exist_ok=True)
        return root
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{make_default_export_stem(config)}.json"


def resolve_viewer_manifest_path(config: Prob3Config) -> Path:
    root = Path(config.viewer_export_path)
    runtime_dir = root if root.suffix != ".json" else root.parent
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / "viewer_manifest.json"
