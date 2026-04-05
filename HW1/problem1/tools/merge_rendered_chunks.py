"""Merge rendered Problem 1 chunks back into a viewer export."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from ..core.config import config_from_render_config
    from ..core.viewer_io import load_snapshot_export, update_viewer_manifest, write_viewer_export
except ImportError:  # pragma: no cover - direct script execution on gx10
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config_from_render_config
    from core.viewer_io import load_snapshot_export, update_viewer_manifest, write_viewer_export


def resolve_default_chunk_glob(snapshot_export_path: Path) -> str:
    return str(snapshot_export_path.parent / "chunks" / f"{snapshot_export_path.stem}_chunk_*.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge rendered Problem 1 chunks into a viewer export.")
    parser.add_argument("--snapshot-export", type=Path, required=True)
    parser.add_argument(
        "--chunk-glob",
        type=str,
        default=None,
        help="Glob pattern for chunk files. Defaults to runtime/chunks/<snapshot>_chunk_*.json",
    )
    parser.add_argument("--viewer-export", type=Path, default=None)
    parser.add_argument(
        "--viewer-manifest",
        type=Path,
        default=Path("hf_space_hw1_problem1") / "runtime" / "viewer_manifest.json",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every snapshot in the snapshot export has a rendered step.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = load_snapshot_export(args.snapshot_export)
    render_config = payload["render_config"]
    config = config_from_render_config(render_config)
    config.run_name = payload.get("run", {}).get("name")

    default_viewer_export = Path(
        payload.get("snapshot_export", {}).get("viewer_export_path", args.snapshot_export.with_name(f"{args.snapshot_export.stem}.json"))
    )
    viewer_export_path = args.viewer_export or default_viewer_export
    config.viewer_export_path = str(viewer_export_path)

    chunk_glob = args.chunk_glob or resolve_default_chunk_glob(args.snapshot_export)
    chunk_paths = [Path(path) for path in sorted(glob.glob(chunk_glob))]
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files matched: {chunk_glob}")

    step_map: dict[int, dict[str, object]] = {}
    for chunk_path in chunk_paths:
        chunk_payload = json.loads(chunk_path.read_text())
        for step in chunk_payload.get("timeline_steps", []):
            step_map[int(step["global_step"])] = step

    timeline_steps = [step_map[key] for key in sorted(step_map)]
    snapshot_count = len(payload.get("timeline_snapshots", []))
    if args.require_complete and len(timeline_steps) != snapshot_count:
        raise ValueError(
            f"Rendered steps are incomplete: got {len(timeline_steps)} of {snapshot_count} snapshots."
        )

    runtime_circuit_path = viewer_export_path.with_name(Path(payload["assets"]["circuit"]).name)
    write_viewer_export(
        config,
        viewer_export_path,
        runtime_circuit_path,
        timeline_steps,
        payload["samples"]["train"],
        payload["samples"]["test"],
    )

    best_test_mse = min((float(step["test_mse"]) for step in timeline_steps), default=None)
    final_train_mse = float(timeline_steps[-1]["train_mse"]) if timeline_steps else None
    final_test_mse = float(timeline_steps[-1]["test_mse"]) if timeline_steps else None
    update_viewer_manifest(
        args.viewer_manifest,
        viewer_export_path,
        config,
        best_test_mse,
        final_train_mse,
        final_test_mse,
        timeline_steps,
    )

    print(f"snapshot_export={args.snapshot_export}")
    print(f"chunks={len(chunk_paths)}")
    print(f"merged_steps={len(timeline_steps)}")
    print(f"viewer_export={viewer_export_path}")
    print(f"viewer_manifest={args.viewer_manifest}")


if __name__ == "__main__":
    main()
