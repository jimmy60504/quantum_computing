"""Merge metrics-only Problem 1 chunks into a standalone summary export."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from ..core.viewer_io import load_snapshot_export
except ImportError:  # pragma: no cover - direct script execution on gx10
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.viewer_io import load_snapshot_export


def resolve_default_chunk_glob(snapshot_export_path: Path) -> str:
    return str(snapshot_export_path.parent / "metrics" / f"{snapshot_export_path.stem}_metrics_chunk_*.json")


def resolve_default_output_path(snapshot_export_path: Path) -> Path:
    stem = snapshot_export_path.stem.removesuffix("_snapshots")
    return snapshot_export_path.with_name(f"{stem}_metrics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge metrics-only Problem 1 chunks into a summary export.")
    parser.add_argument("--snapshot-export", type=Path, required=True)
    parser.add_argument(
        "--chunk-glob",
        type=str,
        default=None,
        help="Glob pattern for metrics chunk files. Defaults to runtime/metrics/<snapshot>_metrics_chunk_*.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every snapshot in the snapshot export has a metrics entry.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = load_snapshot_export(args.snapshot_export)

    chunk_glob = args.chunk_glob or resolve_default_chunk_glob(args.snapshot_export)
    chunk_paths = [Path(path) for path in sorted(glob.glob(chunk_glob))]
    if not chunk_paths:
        raise FileNotFoundError(f"No metrics chunk files matched: {chunk_glob}")

    metric_map: dict[int, dict[str, object]] = {}
    for chunk_path in chunk_paths:
        chunk_payload = json.loads(chunk_path.read_text())
        for metric in chunk_payload.get("timeline_metrics", []):
            metric_map[int(metric["global_step"])] = metric

    timeline_metrics = [metric_map[key] for key in sorted(metric_map)]
    snapshot_count = len(payload.get("timeline_snapshots", []))
    if args.require_complete and len(timeline_metrics) != snapshot_count:
        raise ValueError(
            f"Metrics are incomplete: got {len(timeline_metrics)} of {snapshot_count} snapshots."
        )

    best_test_mse = min((float(step["test_mse"]) for step in timeline_metrics), default=None)
    final_train_mse = float(timeline_metrics[-1]["train_mse"]) if timeline_metrics else None
    final_test_mse = float(timeline_metrics[-1]["test_mse"]) if timeline_metrics else None

    output_path = args.output or resolve_default_output_path(args.snapshot_export)
    output_payload = {
        "title": payload.get("title"),
        "subtitle": payload.get("subtitle"),
        "status": "metrics export",
        "description": (
            "This export stores per-step train/test MSE computed from snapshot states, "
            "without the heavier viewer heatmaps."
        ),
        "run": payload.get("run", {}),
        "snapshot_export": str(args.snapshot_export),
        "summary": {
            "steps": len(timeline_metrics),
            "best_test_mse": best_test_mse,
            "final_train_mse": final_train_mse,
            "final_test_mse": final_test_mse,
        },
        "timeline_metrics": timeline_metrics,
        "loss_history": payload.get("loss_history", []),
    }
    output_path.write_text(json.dumps(output_payload, indent=2) + "\n")

    print(f"snapshot_export={args.snapshot_export}")
    print(f"chunks={len(chunk_paths)}")
    print(f"merged_metrics={len(timeline_metrics)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
