"""Configuration helpers for HW1 Problem 2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HF_SPACE_DIR = Path("HW1") / "problem2" / "hf_space"
RUNTIME_DIR = HF_SPACE_DIR / "runtime"


@dataclass
class Prob2Config:
    seed: int = 11224001
    n_samples: int = 200
    epochs: int = 50
    learning_rate: float = 0.05
    batch_size: int = 32

    num_qubits: int = 2
    num_layers_explicit: int = 4
    num_layers_reuploading: int = 4

    boundary_grid_size: int = 50
    viewer_export_every: int = 5
    viewer_export_path: str = str(RUNTIME_DIR)

    tracking_uri: str = ""
    experiment_name: str = "hw1-problem2-qml-classifiers"
    run_name: str = ""

    device_name: str = "default.qubit"
    diff_method: str = "backprop"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "prob2-run"


def make_default_export_stem(config: Prob2Config) -> str:
    if config.run_name:
        return slugify(config.run_name)
    return slugify(
        f"q{config.num_qubits}-le{config.num_layers_explicit}-"
        f"lr{config.num_layers_reuploading}-e{config.epochs}-n{config.n_samples}"
    )


def resolve_viewer_export_path(config: Prob2Config) -> Path:
    root = Path(config.viewer_export_path)
    if root.suffix == ".json":
        root.parent.mkdir(parents=True, exist_ok=True)
        return root
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{make_default_export_stem(config)}.json"


def resolve_viewer_manifest_path(config: Prob2Config) -> Path:
    root = Path(config.viewer_export_path)
    runtime_dir = root if root.suffix != ".json" else root.parent
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / "viewer_manifest.json"
