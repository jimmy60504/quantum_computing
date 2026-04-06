"""Viewer export helpers for HW1 Problem 3."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Prob3Config


def build_experiment_metadata(config: Prob3Config) -> dict[str, object]:
    return {
        "model": "CNN + PQC hybrid classifier",
        "task": "CIFAR-10 10-class image classification",
        "methods": ["mlp", "qnn"],
        "dataset": "CIFAR-10",
        "device": f"{config.q_device_name} + {config.q_diff_method}",
        "note": "Comparing classical MLP head vs quantum circuit head on shared CNN backbone.",
    }


def build_viewer_payload(
    config: Prob3Config,
    export_path: Path,
    artifacts: dict[str, dict],
) -> dict[str, object]:
    """Build the main viewer JSON payload from collected artifacts."""

    # Build per-method timelines
    methods_data: dict[str, object] = {}
    for method, artifact in artifacts.items():
        methods_data[method] = {
            "label": artifact.get("method_label", method),
            "num_params": artifact.get("num_params", 0),
            "train_time_s": artifact.get("train_time_s", 0),
            "best_test_acc": artifact.get("best_test_acc", 0),
            "history": artifact.get("history", []),
        }

    # Aligned timeline: merge epochs from all methods
    all_epochs = set()
    for artifact in artifacts.values():
        for step in artifact.get("history", []):
            all_epochs.add(step["epoch"])

    timeline_steps: list[dict[str, object]] = []
    for epoch in sorted(all_epochs):
        step: dict[str, object] = {
            "epoch": epoch,
            "global_step": epoch,
            "label": f"Epoch {epoch}",
        }
        # Per-method metrics at this epoch
        for method, artifact in artifacts.items():
            for h in artifact.get("history", []):
                if h["epoch"] == epoch:
                    step[f"{method}_train_loss"] = h.get("train_loss")
                    step[f"{method}_train_acc"] = h.get("train_acc")
                    step[f"{method}_test_loss"] = h.get("test_loss")
                    step[f"{method}_test_acc"] = h.get("test_acc")
                    if "confusion" in h:
                        step[f"{method}_confusion"] = h["confusion"]
                    break
        timeline_steps.append(step)

    # Summary comparison
    summary: dict[str, object] = {}
    for method, artifact in artifacts.items():
        summary[method] = {
            "label": artifact.get("method_label", method),
            "num_params": artifact.get("num_params", 0),
            "train_time_s": artifact.get("train_time_s", 0),
            "best_test_acc": artifact.get("best_test_acc", 0),
        }

    return {
        "title": "Hybrid QNN Explorer",
        "subtitle": "QCAA HW1 Problem 3 — CNN+MLP vs CNN+QNN on CIFAR-10",
        "status": "trajectory export",
        "description": (
            "This export compares a classical MLP classification head against "
            "a parameterized quantum circuit (PQC) head, both built on the same "
            "CNN feature extractor, evaluated on CIFAR-10."
        ),
        "experiment": build_experiment_metadata(config),
        "methods": methods_data,
        "timeline_steps": timeline_steps,
        "summary": summary,
        "run": {
            "name": config.run_name or export_path.stem,
            "path": f"./runtime/{export_path.name}",
        },
    }


def write_viewer_export(
    config: Prob3Config,
    export_path: Path,
    artifacts: dict[str, dict],
) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_viewer_payload(config, export_path, artifacts)
    export_path.write_text(json.dumps(payload, indent=2) + "\n")
    return export_path


def update_viewer_manifest(
    manifest_path: Path,
    export_path: Path,
    config: Prob3Config,
    best_test_acc: float,
    artifacts: dict[str, dict],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"title": "Hybrid QNN Runs", "default_run": None, "runs": []}

    stem = export_path.stem
    label = config.run_name or stem
    entry = {
        "id": stem,
        "label": label,
        "path": f"./runtime/{export_path.name}",
        "epochs": config.epochs,
        "methods": ["mlp", "qnn"],
        "num_qubits": config.num_qubits,
        "num_layers": config.num_layers,
        "best_test_acc": best_test_acc,
    }

    # Add per-method param counts if available
    for method, artifact in artifacts.items():
        entry[f"{method}_params"] = artifact.get("num_params", 0)

    runs = [run for run in manifest.get("runs", []) if run.get("id") != stem]
    runs.insert(0, entry)
    manifest["runs"] = runs
    manifest["default_run"] = stem
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
