"""CLI entry point for HW1 Problem 2 training."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .core.config import Prob2Config
    from .core.training import (
        assemble_viewer_stage,
        prepare_datasets_stage,
        train,
        train_method_stage,
    )
except ImportError:  # pragma: no cover - direct script execution on gx10
    from core.config import Prob2Config
    from core.training import (
        assemble_viewer_stage,
        prepare_datasets_stage,
        train,
        train_method_stage,
    )


# ---------------------------------------------------------------------------
# Shared argument-group helpers
# ---------------------------------------------------------------------------


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=11224001)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-qubits", type=int, default=2)
    p.add_argument("--layers-explicit", type=int, default=4)
    p.add_argument("--layers-reuploading", type=int, default=4)
    p.add_argument("--boundary-grid-size", type=int, default=50)
    p.add_argument("--viewer-export-every", type=int, default=5)
    p.add_argument("--device", type=str, default="default.qubit")
    p.add_argument("--diff-method", type=str, default="backprop")


def _add_viewer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--viewer-export-path", type=str, default="HW1/problem2/hf_space/runtime")


def _add_mlflow_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tracking-uri", type=str, default="")
    p.add_argument("--experiment-name", type=str, default="hw1-problem2-qml-classifiers")
    p.add_argument("--run-name", type=str, default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HW1 Problem 2 QML training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands
-----------
  prepare   Generate + save datasets to --run-dir/datasets.npz
  run       Train one method using the saved dataset
  assemble  Combine method artifacts into the viewer JSON
  all       Full end-to-end pipeline (original behaviour)

Typical staged workflow
-----------------------
  train.py prepare  --run-dir runs/exp1 --n-samples 200
  train.py run      --run-dir runs/exp1 --method explicit  --epochs 30
  train.py run      --run-dir runs/exp1 --method reuploading --epochs 30
  train.py run      --run-dir runs/exp1 --method kernel
  train.py assemble --run-dir runs/exp1
""",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    # -- prepare --
    p_prep = sub.add_parser("prepare", help="Generate and save datasets.")
    p_prep.add_argument("--run-dir", required=True, help="Directory to write datasets.npz.")
    _add_data_args(p_prep)
    _add_mlflow_args(p_prep)

    # -- run --
    p_run = sub.add_parser("run", help="Train one QML method using saved dataset.")
    p_run.add_argument("--run-dir", required=True, help="Directory containing datasets.npz.")
    p_run.add_argument(
        "--method",
        required=True,
        choices=["explicit", "reuploading", "kernel"],
        help="Which method to train.",
    )
    p_run.add_argument("--seed", type=int, default=11224001)
    _add_model_args(p_run)
    _add_viewer_args(p_run)
    _add_mlflow_args(p_run)

    # -- assemble --
    p_asm = sub.add_parser("assemble", help="Build viewer JSON from all method artifacts.")
    p_asm.add_argument("--run-dir", required=True, help="Directory containing *_artifact.json.")
    p_asm.add_argument("--seed", type=int, default=11224001)
    _add_viewer_args(p_asm)
    _add_mlflow_args(p_asm)

    # -- all (original end-to-end behaviour) --
    p_all = sub.add_parser("all", help="Run the full pipeline end-to-end.")
    _add_data_args(p_all)
    _add_model_args(p_all)
    _add_viewer_args(p_all)
    _add_mlflow_args(p_all)

    return parser


def _make_config(args: argparse.Namespace) -> Prob2Config:
    return Prob2Config(
        seed=getattr(args, "seed", 11224001),
        n_samples=getattr(args, "n_samples", 200),
        epochs=getattr(args, "epochs", 50),
        learning_rate=getattr(args, "learning_rate", 0.05),
        batch_size=getattr(args, "batch_size", 32),
        num_qubits=getattr(args, "num_qubits", 2),
        num_layers_explicit=getattr(args, "layers_explicit", 4),
        num_layers_reuploading=getattr(args, "layers_reuploading", 4),
        boundary_grid_size=getattr(args, "boundary_grid_size", 50),
        viewer_export_every=max(1, getattr(args, "viewer_export_every", 5)),
        viewer_export_path=getattr(args, "viewer_export_path", "HW1/problem2/hf_space/runtime"),
        tracking_uri=getattr(args, "tracking_uri", ""),
        experiment_name=getattr(args, "experiment_name", "hw1-problem2-qml-classifiers"),
        run_name=getattr(args, "run_name", ""),
        device_name=getattr(args, "device", "default.qubit"),
        diff_method=getattr(args, "diff_method", "backprop"),
    )


def main() -> None:
    args = build_parser().parse_args()
    config = _make_config(args)

    if args.stage == "prepare":
        prepare_datasets_stage(config, Path(args.run_dir))
    elif args.stage == "run":
        train_method_stage(args.method, config, Path(args.run_dir))
    elif args.stage == "assemble":
        assemble_viewer_stage(config, Path(args.run_dir))
    elif args.stage == "all":
        train(config)
    else:
        raise ValueError(f"Unknown stage: {args.stage!r}")


if __name__ == "__main__":
    main()
