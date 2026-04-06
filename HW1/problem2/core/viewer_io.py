"""Viewer export helpers for HW1 Problem 2."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Prob2Config


def build_experiment_metadata(config: Prob2Config) -> dict[str, object]:
    return {
        "model": "PennyLane QML classifiers",
        "task": "Binary classification on circle and moons datasets",
        "methods": ["explicit", "kernel", "reuploading"],
        "datasets": ["circle", "moons"],
        "device": f"{config.device_name} + {config.diff_method}",
        "note": "Slider steps correspond to exported training epochs for the trainable models.",
    }


def build_viewer_payload(
    config: Prob2Config,
    viewer_export_path: Path,
    timeline_steps: list[dict[str, object]],
    scatter: dict[str, list[dict[str, object]]],
    summary: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "QML Classifier Explorer",
        "subtitle": "QCAA HW1 Problem 2 classification results",
        "status": "trajectory export",
        "description": (
            "This export compares explicit, kernel, and data-reuploading quantum "
            "classifiers across the circle and moons datasets."
        ),
        "experiment": build_experiment_metadata(config),
        "assets": {
            "datasets_overview": "assets/preview_datasets.png",
        },
        "grid": {
            "resolution": config.boundary_grid_size,
        },
        "scatter": scatter,
        "timeline_steps": timeline_steps,
        "run": {
            "name": config.run_name or viewer_export_path.stem,
            "path": f"./runtime/{viewer_export_path.name}",
        },
    }
    if summary is not None:
        payload["summary"] = summary
    return payload


def write_viewer_export(
    config: Prob2Config,
    viewer_export_path: Path,
    timeline_steps: list[dict[str, object]],
    scatter: dict[str, list[dict[str, object]]],
    summary: dict[str, object] | None = None,
    use_chunks: bool = False,
) -> Path:
    """Write the viewer export JSON.

    When *use_chunks* is True the heavy boundary data is split into per-epoch
    chunk files under ``<export_dir>/chunks/``.  Each timeline step in the main
    JSON then carries a ``chunk_path`` pointer instead of inline ``boundaries``.
    The main JSON also gets a ``timeline_chunks`` list used by the frontend for
    background prefetching.

    When *use_chunks* is False (default, backward-compatible) all data is
    written inline into a single file.
    """
    viewer_export_path.parent.mkdir(parents=True, exist_ok=True)

    if use_chunks:
        chunks_dir = viewer_export_path.parent / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        slim_steps: list[dict[str, object]] = []
        chunk_index: list[dict[str, object]] = []

        for step in timeline_steps:
            epoch = int(step.get("epoch", step.get("global_step", 0)))
            chunk_name = f"{viewer_export_path.stem}_epoch_{epoch:04d}.json"
            chunk_path_abs = chunks_dir / chunk_name
            chunk_rel_path = f"./runtime/chunks/{chunk_name}"

            # Write chunk: only the heavy boundary data keyed by epoch
            chunk_payload: dict[str, object] = {
                "timeline_steps": [
                    {
                        "epoch": epoch,
                        "global_step": step.get("global_step", epoch),
                        "boundaries": step.get("boundaries"),
                    }
                ]
            }
            chunk_path_abs.write_text(json.dumps(chunk_payload) + "\n")

            # Slim step: metrics only + chunk pointer
            slim_step = {k: v for k, v in step.items() if k not in ("boundaries", "scatter")}
            slim_step["chunk_path"] = chunk_rel_path
            slim_steps.append(slim_step)
            chunk_index.append({"path": chunk_rel_path})

        payload = build_viewer_payload(config, viewer_export_path, slim_steps, scatter, summary)
        payload["timeline_chunks"] = chunk_index
    else:
        payload = build_viewer_payload(config, viewer_export_path, timeline_steps, scatter, summary)

    viewer_export_path.write_text(json.dumps(payload, indent=2) + "\n")
    return viewer_export_path


def update_viewer_manifest(
    manifest_path: Path,
    export_path: Path,
    config: Prob2Config,
    best_test_acc: float | None,
    timeline_steps: list[dict[str, object]],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"title": "QML Classifier Runs", "default_run": None, "runs": []}

    stem = export_path.stem
    label = config.run_name or stem
    entry = {
        "id": stem,
        "label": label,
        "path": f"./runtime/{export_path.name}",
        "steps": len(timeline_steps),
        "method": "explicit + kernel + reuploading",
        "dataset": "circle + moons",
        "methods": ["explicit", "kernel", "reuploading"],
        "datasets": ["circle", "moons"],
        "num_qubits": config.num_qubits,
        "num_layers": max(config.num_layers_explicit, config.num_layers_reuploading),
        "num_layers_explicit": config.num_layers_explicit,
        "num_layers_reuploading": config.num_layers_reuploading,
        "best_test_acc": best_test_acc,
    }

    runs = [run for run in manifest.get("runs", []) if run.get("id") != stem]
    runs.insert(0, entry)
    manifest["runs"] = runs
    manifest["default_run"] = stem
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
