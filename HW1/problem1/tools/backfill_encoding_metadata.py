"""Backfill encoding metadata for HW1 Problem 1 viewer exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ENCODINGS = ("raw", "poly", "exp")


def infer_encoding(identifier: str | None, fallback: str) -> str:
    text = (identifier or "").lower()
    for encoding in ENCODINGS:
        if text.startswith(f"{encoding}-") or f"-{encoding}-" in text or text == encoding:
            return encoding
    return fallback


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def backfill_export(export_path: Path, default_encoding: str) -> tuple[bool, str]:
    payload = load_json(export_path)
    experiment = payload.setdefault("experiment", {})
    run = payload.get("run", {})
    inferred = infer_encoding(run.get("name") or export_path.stem, default_encoding)
    previous = experiment.get("encoding")
    if previous == inferred:
        return False, inferred
    experiment["encoding"] = inferred
    dump_json(export_path, payload)
    return True, inferred


def backfill_manifest(manifest_path: Path, default_encoding: str) -> tuple[bool, list[tuple[str, str]]]:
    payload = load_json(manifest_path)
    changed = False
    updates: list[tuple[str, str]] = []

    for run in payload.get("runs", []):
        inferred = infer_encoding(run.get("id") or run.get("label"), default_encoding)
        if run.get("encoding_mode") != inferred:
            run["encoding_mode"] = inferred
            changed = True
        updates.append((run.get("id", "unknown"), inferred))

    if changed:
        dump_json(manifest_path, payload)

    return changed, updates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill encoding metadata into viewer JSON files.")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("hf_space_hw1_problem1") / "runtime",
        help="Directory that contains viewer exports and viewer_manifest.json.",
    )
    parser.add_argument(
        "--default-encoding",
        type=str,
        choices=ENCODINGS,
        default="raw",
        help="Encoding to use when an old run name does not encode the mode explicitly.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime_dir = args.runtime_dir
    manifest_path = runtime_dir / "viewer_manifest.json"

    if not runtime_dir.exists():
        raise FileNotFoundError(f"Runtime directory not found: {runtime_dir}")

    export_updates: list[tuple[str, str]] = []
    for export_path in sorted(runtime_dir.glob("*.json")):
        if export_path.name == "viewer_manifest.json" or export_path.name.endswith("_fourier_summary.json"):
            continue
        changed, encoding = backfill_export(export_path, args.default_encoding)
        if changed:
            export_updates.append((export_path.name, encoding))

    if manifest_path.exists():
        manifest_changed, manifest_updates = backfill_manifest(manifest_path, args.default_encoding)
    else:
        manifest_changed, manifest_updates = False, []

    print(f"runtime_dir={runtime_dir}")
    print(f"default_encoding={args.default_encoding}")
    print(f"exports_updated={len(export_updates)}")
    for filename, encoding in export_updates:
        print(f"  export {filename} -> encoding={encoding}")
    print(f"manifest_updated={manifest_changed}")
    for run_id, encoding in manifest_updates:
        print(f"  manifest {run_id} -> encoding_mode={encoding}")


if __name__ == "__main__":
    main()
