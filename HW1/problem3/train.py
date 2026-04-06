"""CLI entry point for HW1 Problem 3 training."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .core.config import Prob3Config
    from .core.training import train_method, assemble_viewer, train_all
except ImportError:
    from core.config import Prob3Config
    from core.training import train_method, assemble_viewer, train_all


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--num-qubits", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--viewer-export-every", type=int, default=1)
    p.add_argument("--q-device", type=str, default="default.qubit")
    p.add_argument("--q-diff-method", type=str, default="backprop")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])


def _add_viewer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--viewer-export-path", type=str, default="HW1/problem3/hf_space/runtime")


def _add_mlflow_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tracking-uri", type=str, default="")
    p.add_argument("--experiment-name", type=str, default="hw1-problem3-hybrid-qnn")
    p.add_argument("--run-name", type=str, default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HW1 Problem 3 — Hybrid CNN+QNN training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands
-----------
  run       Train one method (mlp or qnn)
  assemble  Combine method artifacts into the viewer JSON
  all       Full end-to-end pipeline (train both + assemble)

Typical staged workflow
-----------------------
  train.py run      --run-dir runs/exp1 --method mlp   --epochs 20
  train.py run      --run-dir runs/exp1 --method qnn   --epochs 20
  train.py assemble --run-dir runs/exp1
""",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Train one method (mlp or qnn).")
    p_run.add_argument("--run-dir", required=True, help="Directory for artifacts.")
    p_run.add_argument(
        "--method", required=True, choices=["mlp", "qnn"],
        help="Which method to train.",
    )
    p_run.add_argument("--seed", type=int, default=11224001)
    _add_model_args(p_run)
    _add_viewer_args(p_run)
    _add_mlflow_args(p_run)

    # -- assemble --
    p_asm = sub.add_parser("assemble", help="Build viewer JSON from artifacts.")
    p_asm.add_argument("--run-dir", required=True)
    p_asm.add_argument("--seed", type=int, default=11224001)
    _add_viewer_args(p_asm)
    _add_mlflow_args(p_asm)

    # -- all --
    p_all = sub.add_parser("all", help="Full pipeline: train both methods + assemble.")
    p_all.add_argument("--seed", type=int, default=11224001)
    _add_model_args(p_all)
    _add_viewer_args(p_all)
    _add_mlflow_args(p_all)

    return parser


def _make_config(args: argparse.Namespace) -> Prob3Config:
    return Prob3Config(
        seed=getattr(args, "seed", 11224001),
        epochs=getattr(args, "epochs", 20),
        learning_rate=getattr(args, "learning_rate", 1e-3),
        batch_size=getattr(args, "batch_size", 64),
        feature_dim=getattr(args, "feature_dim", 256),
        num_qubits=getattr(args, "num_qubits", 8),
        num_layers=getattr(args, "num_layers", 4),
        freeze_backbone=getattr(args, "freeze_backbone", False),
        viewer_export_every=max(1, getattr(args, "viewer_export_every", 1)),
        viewer_export_path=getattr(args, "viewer_export_path", "HW1/problem3/hf_space/runtime"),
        q_device_name=getattr(args, "q_device", "default.qubit"),
        q_diff_method=getattr(args, "q_diff_method", "backprop"),
        device=getattr(args, "device", "cpu"),
        tracking_uri=getattr(args, "tracking_uri", ""),
        experiment_name=getattr(args, "experiment_name", "hw1-problem3-hybrid-qnn"),
        run_name=getattr(args, "run_name", ""),
    )


def main() -> None:
    args = build_parser().parse_args()
    config = _make_config(args)

    if args.stage == "run":
        train_method(args.method, config, Path(args.run_dir))
    elif args.stage == "assemble":
        assemble_viewer(config, Path(args.run_dir))
    elif args.stage == "all":
        train_all(config)
    else:
        raise ValueError(f"Unknown stage: {args.stage!r}")


if __name__ == "__main__":
    main()
