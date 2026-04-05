"""Viewer export and manifest helpers for HW1 Problem 1."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .config import Config
except ImportError:  # pragma: no cover - direct script execution on gx10
    from config import Config


def build_experiment_metadata(config: Config, note: str | None = None) -> dict[str, str]:
    return {
        "model": "PennyLane data reuploading regressor",
        "task": "Regression on f(x1, x2) = sin(exp(x1) + x2)",
        "encoding": config.encoding_mode,
        "train_domain": "[0.0, 0.5] x [0.0, 0.5]",
        "test_domain": "[0.5, 1.0] x [0.5, 1.0]",
        "device": f"{config.device_name} + {config.diff_method}",
        "note": note or "Slider steps correspond to exported training batches.",
    }


def build_viewer_payload(
    config: Config,
    viewer_export_path: Path,
    runtime_circuit_path: Path,
    timeline_steps: list[dict[str, object]],
    train_points: dict[str, list[float]],
    test_points: dict[str, list[float]],
    timeline_chunks: list[dict[str, object]] | None = None,
    status: str = "trajectory export",
    description: str | None = None,
) -> dict[str, object]:
    payload = {
        "title": "Data Reuploading Explorer",
        "subtitle": "QCAA HW1 Problem 1 regression results",
        "status": status,
        "description": description
        or (
            "This export is generated from the latest training run and stores raw heatmap "
            "grids per batch step for Plotly playback."
        ),
        "experiment": build_experiment_metadata(config),
        "assets": {
            "circuit": f"runtime/{runtime_circuit_path.name}",
            "data_overview": "assets/problem1_data_overview.png",
        },
        "grid": {
            "grid_size": config.heatmap_grid_size,
        },
        "samples": {
            "train": train_points,
            "test": test_points,
        },
        "timeline_steps": timeline_steps,
        "run": {
            "name": config.run_name or viewer_export_path.stem,
            "path": f"./runtime/{viewer_export_path.name}",
        },
    }
    if timeline_chunks:
        payload["timeline_chunks"] = timeline_chunks
    return payload


def write_snapshot_export(
    config: Config,
    snapshot_export_path: Path,
    viewer_export_path: Path,
    runtime_circuit_path: Path,
    train_points: dict[str, list[float]],
    test_points: dict[str, list[float]],
    timeline_snapshots: list[dict[str, object]],
    loss_history: list[dict[str, float]],
    num_samples: int,
) -> Path:
    snapshot_export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_viewer_payload(
        config,
        viewer_export_path,
        runtime_circuit_path,
        timeline_steps=[],
        train_points=train_points,
        test_points=test_points,
        status="snapshot export",
        description=(
            "This export stores per-step model snapshots so rendering can be offloaded "
            "to other machines and merged later."
        ),
    )
    payload["render_config"] = {
        "num_qubits": config.num_qubits,
        "num_layers": config.num_layers,
        "encoding_mode": config.encoding_mode,
        "hidden_scale": config.hidden_scale,
        "device_name": config.device_name,
        "diff_method": config.diff_method,
        "heatmap_grid_size": config.heatmap_grid_size,
        "batch_size": config.batch_size,
        "num_samples": num_samples,
        "viewer_export_every": config.viewer_export_every,
    }
    payload["timeline_snapshots"] = timeline_snapshots
    payload["loss_history"] = loss_history
    payload["snapshot_export"] = {
        "path": str(snapshot_export_path),
        "viewer_export_path": str(viewer_export_path),
    }
    snapshot_export_path.write_text(json.dumps(payload, indent=2))
    return snapshot_export_path


def write_viewer_export(
    config: Config,
    viewer_export_path: Path,
    runtime_circuit_path: Path,
    timeline_steps: list[dict[str, object]],
    train_points: dict[str, list[float]],
    test_points: dict[str, list[float]],
    timeline_chunks: list[dict[str, object]] | None = None,
) -> Path:
    viewer_export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_viewer_payload(
        config,
        viewer_export_path,
        runtime_circuit_path,
        timeline_steps,
        train_points,
        test_points,
        timeline_chunks=timeline_chunks,
    )
    viewer_export_path.write_text(json.dumps(payload, indent=2))
    return viewer_export_path


def update_viewer_manifest(
    manifest_path: Path,
    export_path: Path,
    config: Config,
    best_test_mse: float | None,
    final_train_mse: float | None,
    final_test_mse: float | None,
    timeline_steps: list[dict[str, object]],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"title": "Data Reuploading Runs", "default_run": None, "runs": []}

    stem = export_path.stem
    label = config.run_name or stem
    entry = {
        "id": stem,
        "label": label,
        "path": f"./runtime/{export_path.name}",
        "steps": len(timeline_steps),
        "device": config.device_name,
        "diff_method": config.diff_method,
        "num_qubits": config.num_qubits,
        "num_layers": config.num_layers,
        "encoding_mode": config.encoding_mode,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "best_test_mse": best_test_mse,
        "final_train_mse": final_train_mse,
        "final_test_mse": final_test_mse,
    }

    runs = [run for run in manifest.get("runs", []) if run.get("id") != stem]
    runs.insert(0, entry)
    manifest["runs"] = runs
    manifest["default_run"] = stem
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def load_snapshot_export(snapshot_export_path: Path) -> dict[str, object]:
    return json.loads(snapshot_export_path.read_text())
