"""Configuration and path helpers for HW1 Problem 1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAIN_RANGES = np.array([0.0, 0.5] * 2).reshape(2, 2)
TEST_RANGES = np.array([0.5, 1.0] * 2).reshape(2, 2)
ENCODING_CHOICES = (
    "raw",
    "poly",
    "exp",
    "exp_aug",
    "exp_sum",
    "oracle_shortcut",
    "quantum_exact",
)
INPUT_ACTIVATION_CHOICES = ("tanh", "identity")
LR_SCHEDULER_CHOICES = ("none", "cosine")
RENDER_MODE_CHOICES = ("inline", "snapshots-only")


@dataclass
class Config:
    num_qubits: int = 2
    num_layers: int = 3
    encoding_mode: str = "raw"
    render_mode: str = "inline"
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.03
    lr_scheduler: str = "none"
    min_learning_rate: float = 0.0
    hidden_scale: float = 1.0
    input_activation: str = "tanh"
    angle_scale: float = 1.0
    heatmap_grid_size: int = 64
    device_name: str = "lightning.qubit"
    diff_method: str | None = None
    viewer_export_path: str | None = None
    snapshot_export_path: str | None = None
    viewer_export_every: int = 1
    render_workers: int = 4
    tracking_uri: str | None = None
    experiment_name: str = "hw1-problem1-datareuploading"
    run_name: str | None = None


def resolve_diff_method(device_name: str, requested_diff_method: str | None) -> str:
    if requested_diff_method is not None:
        return requested_diff_method

    if device_name == "default.qubit":
        return "backprop"

    if device_name == "lightning.qubit":
        return "adjoint"

    raise ValueError(f"Unsupported device: {device_name}")


def validate_device_config(device_name: str, diff_method: str) -> None:
    supported_pairs = {
        ("default.qubit", "backprop"),
        ("default.qubit", "adjoint"),
        ("lightning.qubit", "adjoint"),
    }
    if (device_name, diff_method) not in supported_pairs:
        supported_text = ", ".join(
            f"{device} + {method}" for device, method in sorted(supported_pairs)
        )
        raise ValueError(
            f"Unsupported device/diff combination: {device_name} + {diff_method}. "
            f"Try one of: {supported_text}."
        )


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "run"


def make_default_export_stem(config: Config) -> str:
    if config.run_name:
        return slugify(config.run_name)

    return slugify(
        (
            f"{config.encoding_mode}-q{config.num_qubits}-l{config.num_layers}-"
            f"act{config.input_activation}-as{config.angle_scale:g}-"
            f"sched{config.lr_scheduler}-minlr{config.min_learning_rate:g}-"
            f"{config.device_name}-{config.diff_method}-"
            f"lr{config.learning_rate}-b{config.batch_size}-e{config.epochs}"
        )
    )


def resolve_viewer_export_path(config: Config) -> Path:
    if config.viewer_export_path:
        return Path(config.viewer_export_path)
    return Path("hf_space_hw1_problem1") / "runtime" / f"{make_default_export_stem(config)}.json"


def resolve_snapshot_export_path(config: Config, viewer_export_path: Path | None = None) -> Path:
    if config.snapshot_export_path:
        return Path(config.snapshot_export_path)

    resolved_viewer_export_path = viewer_export_path or resolve_viewer_export_path(config)
    return resolved_viewer_export_path.with_name(f"{resolved_viewer_export_path.stem}_snapshots.json")


def resolve_runtime_circuit_path(viewer_export_path: Path) -> Path:
    return viewer_export_path.with_name(f"{viewer_export_path.stem}_circuit.png")


def config_from_render_config(render_config: dict[str, object]) -> Config:
    return Config(
        num_qubits=int(render_config["num_qubits"]),
        num_layers=int(render_config["num_layers"]),
        encoding_mode=str(render_config["encoding_mode"]),
        lr_scheduler=str(render_config.get("lr_scheduler", "none")),
        min_learning_rate=float(render_config.get("min_learning_rate", 0.0)),
        hidden_scale=float(render_config["hidden_scale"]),
        input_activation=str(render_config.get("input_activation", "tanh")),
        angle_scale=float(render_config.get("angle_scale", 1.0)),
        heatmap_grid_size=int(render_config["heatmap_grid_size"]),
        batch_size=int(render_config["batch_size"]),
        device_name=str(render_config["device_name"]),
        diff_method=str(render_config["diff_method"]),
    )
