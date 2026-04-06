"""Fourier spectrum analysis for QCAA HW1 Problem 1 viewer exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def resolve_default_export_path() -> Path:
    manifest_path = Path("HW1") / "problem1" / "hf_space" / "runtime" / "viewer_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Could not find HW1/problem1/hf_space/runtime/viewer_manifest.json. "
            "Pass --viewer-export explicitly or run on gx10 where the runtime exports live."
        )

    manifest = json.loads(manifest_path.read_text())
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError("viewer_manifest.json does not contain any runs.")

    default_run = manifest.get("default_run")
    selected = None
    for run in runs:
        if run.get("id") == default_run:
            selected = run
            break
    if selected is None:
        selected = runs[0]

    run_path = selected.get("path")
    if not run_path:
        raise ValueError("Selected run is missing a path in viewer_manifest.json.")

    return manifest_path.parent / Path(run_path).name


def load_test_heatmaps(viewer_export_path: Path) -> tuple[dict, dict]:
    payload = json.loads(viewer_export_path.read_text())
    timeline_steps = payload.get("timeline_steps", [])
    if not timeline_steps:
        raise ValueError(
            f"{viewer_export_path} does not contain timeline_steps, so there is no final heatmap."
        )

    final_step = timeline_steps[-1]
    test_heatmaps = final_step.get("heatmaps", {}).get("test")
    if not test_heatmaps and final_step.get("chunk_path"):
        raw_chunk_path = Path(final_step["chunk_path"])
        candidate_paths = []
        if raw_chunk_path.is_absolute():
            candidate_paths.append(raw_chunk_path)
        else:
            candidate_paths.append((viewer_export_path.parent.parent / raw_chunk_path).resolve())
            candidate_paths.append((viewer_export_path.parent / raw_chunk_path.name).resolve())
            candidate_paths.append(raw_chunk_path.resolve())
        chunk_path = next((path for path in candidate_paths if path.exists()), None)
        if chunk_path is None:
            raise FileNotFoundError(
                f"{viewer_export_path} points to missing chunk file {raw_chunk_path}."
            )
        chunk_payload = json.loads(chunk_path.read_text())
        chunk_steps = chunk_payload.get("timeline_steps", [])
        if not chunk_steps:
            raise ValueError(f"{chunk_path} does not contain any timeline steps.")
        final_step = chunk_steps[-1]
        test_heatmaps = final_step.get("heatmaps", {}).get("test")
    if not test_heatmaps:
        raise ValueError(f"{viewer_export_path} is missing test heatmaps in the final timeline step.")

    return payload, test_heatmaps


def compute_spectrum(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    spectrum = np.fft.fftshift(np.fft.fft2(grid))
    magnitude = np.abs(spectrum)
    freq_x = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(len(y), d=dy))
    return freq_x, freq_y, magnitude


def top_frequency_components(
    magnitude: np.ndarray,
    freq_x: np.ndarray,
    freq_y: np.ndarray,
    top_k: int = 8,
) -> list[dict[str, float]]:
    working = magnitude.copy()
    center_y = working.shape[0] // 2
    center_x = working.shape[1] // 2
    working[center_y, center_x] = 0.0

    flat_indices = np.argsort(working.ravel())[::-1][:top_k]
    components = []
    for flat_index in flat_indices:
        iy, ix = np.unravel_index(int(flat_index), working.shape)
        components.append(
            {
                "fx": float(freq_x[ix]),
                "fy": float(freq_y[iy]),
                "magnitude": float(working[iy, ix]),
            }
        )
    return components


def make_figure(
    target_grid: np.ndarray,
    prediction_grid: np.ndarray,
    target_magnitude: np.ndarray,
    prediction_magnitude: np.ndarray,
    freq_x: np.ndarray,
    freq_y: np.ndarray,
    output_path: Path,
    run_label: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    panels = [
        (
            axes[0, 0],
            target_grid,
            "Target Output on Test Domain",
            "viridis",
            (0.5, 1.0, 0.5, 1.0),
            "x1",
            "x2",
        ),
        (
            axes[0, 1],
            prediction_grid,
            "Model Output on Test Domain",
            "viridis",
            (0.5, 1.0, 0.5, 1.0),
            "x1",
            "x2",
        ),
        (
            axes[1, 0],
            np.log1p(target_magnitude),
            "Target Fourier Spectrum (log magnitude)",
            "magma",
            (freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]),
            "freq x1",
            "freq x2",
        ),
        (
            axes[1, 1],
            np.log1p(prediction_magnitude),
            "Model Fourier Spectrum (log magnitude)",
            "magma",
            (freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]),
            "freq x1",
            "freq x2",
        ),
    ]

    for ax, grid, title, cmap, extent, xlabel, ylabel in panels:
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, extent=extent)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"HW1 Problem 1 Fourier Analysis - {run_label}", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fourier spectrum analysis for HW1 Problem 1.")
    parser.add_argument(
        "--viewer-export",
        type=Path,
        default=None,
        help="Path to a Problem 1 viewer export JSON. Defaults to the viewer manifest default run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the Fourier analysis outputs. Defaults to the viewer export directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many dominant non-DC Fourier components to report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    viewer_export_path = args.viewer_export or resolve_default_export_path()
    payload, test_heatmaps = load_test_heatmaps(viewer_export_path)
    output_dir = args.output_dir or viewer_export_path.parent

    run_label = payload.get("run", {}).get("name", viewer_export_path.stem)
    target = test_heatmaps["target"]
    prediction = test_heatmaps["prediction"]

    x = np.asarray(target["x"], dtype=np.float32)
    y = np.asarray(target["y"], dtype=np.float32)
    target_grid = np.asarray(target["z"], dtype=np.float32)
    prediction_grid = np.asarray(prediction["z"], dtype=np.float32)

    freq_x, freq_y, target_magnitude = compute_spectrum(target_grid, x, y)
    _, _, prediction_magnitude = compute_spectrum(prediction_grid, x, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{viewer_export_path.stem}_fourier_spectrum.png"
    summary_path = output_dir / f"{viewer_export_path.stem}_fourier_summary.json"

    make_figure(
        target_grid,
        prediction_grid,
        target_magnitude,
        prediction_magnitude,
        freq_x,
        freq_y,
        figure_path,
        run_label,
    )

    summary = {
        "run": run_label,
        "viewer_export": str(viewer_export_path),
        "dominant_target_components": top_frequency_components(
            target_magnitude,
            freq_x,
            freq_y,
            top_k=args.top_k,
        ),
        "dominant_model_components": top_frequency_components(
            prediction_magnitude,
            freq_x,
            freq_y,
            top_k=args.top_k,
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"viewer_export={viewer_export_path}")
    print(f"figure={figure_path}")
    print(f"summary={summary_path}")
    print("top target components:")
    for component in summary["dominant_target_components"]:
        print(
            f"  fx={component['fx']:.4f} fy={component['fy']:.4f} "
            f"mag={component['magnitude']:.6f}"
        )
    print("top model components:")
    for component in summary["dominant_model_components"]:
        print(
            f"  fx={component['fx']:.4f} fy={component['fy']:.4f} "
            f"mag={component['magnitude']:.6f}"
        )


if __name__ == "__main__":
    main()
