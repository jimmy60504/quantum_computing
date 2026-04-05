"""Render a slice of Problem 1 snapshots into a standalone chunk JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from ..core.config import config_from_render_config, validate_device_config
    from ..core.modeling import render_timeline_snapshots_parallel
    from ..core.viewer_io import load_snapshot_export
except ImportError:  # pragma: no cover - direct script execution on gx10
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config_from_render_config, validate_device_config
    from core.modeling import render_timeline_snapshots_parallel
    from core.viewer_io import load_snapshot_export


def resolve_default_output_path(snapshot_export_path: Path, start_index: int, end_index: int) -> Path:
    chunk_dir = snapshot_export_path.parent / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    end_label = max(start_index, end_index - 1)
    return chunk_dir / f"{snapshot_export_path.stem}_chunk_{start_index:05d}_{end_label:05d}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a subset of Problem 1 snapshots.")
    parser.add_argument("--snapshot-export", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end index. Defaults to the end of the snapshot list.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--render-workers", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("default.qubit", "lightning.qubit"),
        help="Optional override for the render simulator.",
    )
    parser.add_argument(
        "--diff-method",
        type=str,
        default=None,
        choices=("backprop", "adjoint"),
        help="Optional override for the render differentiation mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = load_snapshot_export(args.snapshot_export)
    snapshots = payload.get("timeline_snapshots", [])

    start_index = max(0, args.start_index)
    end_index = len(snapshots) if args.end_index is None else min(len(snapshots), args.end_index)
    if start_index >= end_index:
        raise ValueError(f"Empty snapshot slice: start={start_index} end={end_index}")

    config = config_from_render_config(payload["render_config"])
    if args.device is not None:
        config.device_name = args.device
    if args.diff_method is not None:
        config.diff_method = args.diff_method
    validate_device_config(config.device_name, config.diff_method or "adjoint")
    config.render_workers = args.render_workers or max(1, min(len(snapshots[start_index:end_index]), os.cpu_count() or 1))

    rendered_steps = render_timeline_snapshots_parallel(
        config,
        int(payload["render_config"]["num_samples"]),
        snapshots[start_index:end_index],
    )

    output_path = args.output or resolve_default_output_path(args.snapshot_export, start_index, end_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_payload = {
        "snapshot_export": str(args.snapshot_export),
        "run": payload.get("run", {}),
        "range": {
            "start_index": start_index,
            "end_index": end_index,
            "count": len(rendered_steps),
        },
        "render_config": {
            "device_name": config.device_name,
            "diff_method": config.diff_method,
            "render_workers": config.render_workers,
        },
        "timeline_steps": rendered_steps,
    }
    output_path.write_text(json.dumps(chunk_payload, indent=2) + "\n")

    print(f"snapshot_export={args.snapshot_export}")
    print(f"output={output_path}")
    print(f"steps_rendered={len(rendered_steps)}")
    print(f"range={start_index}:{end_index}")


if __name__ == "__main__":
    main()
