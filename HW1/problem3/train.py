"""CLI entry point for HW1 Problem 3 training."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .core.config import Prob3Config
    from .core.training import train_method, assemble_viewer, train_all
    from .core.probing import export_gallery, probe_checkpoints, probe_tsne
except ImportError:
    from core.config import Prob3Config
    from core.training import train_method, assemble_viewer, train_all
    from core.probing import export_gallery, probe_checkpoints, probe_tsne


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature-dim", type=int, default=256)
    p.add_argument("--num-qubits", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--checkpoint-freq", type=int, default=0,
                   help="Checkpoints per epoch across ALL epochs (0=off).")
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
  gallery   Export candidate test images for manual selection
  probe     Run selected samples through saved checkpoints
  tsne      Extract backbone features at checkpoints and run t-SNE/UMAP

Typical staged workflow
-----------------------
  train.py run      --run-dir runs/exp1 --method mlp   --epochs 20 --checkpoint-epochs 1
  train.py run      --run-dir runs/exp1 --method qnn   --epochs 20 --checkpoint-epochs 1
  train.py gallery  --run-dir runs/exp1
  # ... pick images from runs/exp1/gallery/ ...
  train.py probe    --run-dir runs/exp1 --samples 49,15,78,33,5,62,11,97,1,44
  train.py tsne     --run-dir runs/exp1 --n-per-class 20 --n-steps 50
  train.py assemble --run-dir runs/exp1
""",
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    # -- run --
    p_run = sub.add_parser("run", help="Train one method (mlp or qnn).")
    p_run.add_argument("--run-dir", required=True, help="Directory for artifacts.")
    p_run.add_argument(
        "--method", required=True, choices=["mlp", "qnn", "mlp_unf"],
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

    # -- gallery --
    p_gal = sub.add_parser("gallery", help="Export candidate test images for selection.")
    p_gal.add_argument("--run-dir", required=True)
    p_gal.add_argument("--per-class", type=int, default=10,
                       help="Number of candidate images per class.")
    p_gal.add_argument("--data-dir", type=str, default="./data/cifar10")

    # -- probe --
    p_prb = sub.add_parser("probe", help="Probe checkpoints with selected samples.")
    p_prb.add_argument("--run-dir", required=True)
    p_prb.add_argument("--samples", type=str, default=None,
                       help="Comma-separated dataset indices (e.g. 49,15,78).")
    p_prb.add_argument("--methods", type=str, default=None,
                       help="Comma-separated methods to probe (e.g. mlp,qnn). Default: all.")
    p_prb.add_argument("--data-dir", type=str, default="./data/cifar10")
    p_prb.add_argument("--seed", type=int, default=11224001)
    _add_model_args(p_prb)

    # -- tsne --
    p_tsn = sub.add_parser("tsne", help="Extract backbone features + run t-SNE/UMAP.")
    p_tsn.add_argument("--run-dir", required=True)
    p_tsn.add_argument("--n-per-class", type=int, default=20,
                       help="Number of test samples per class (default 20 → 200 total).")
    p_tsn.add_argument("--n-steps", type=int, default=50,
                       help="Number of evenly-spaced checkpoint frames (default 50).")
    p_tsn.add_argument("--reduction", type=str, default="tsne",
                       choices=["tsne", "umap", "pca"],
                       help="Dimensionality reduction method (default tsne).")
    p_tsn.add_argument("--methods", type=str, default=None,
                       help="Comma-separated methods (e.g. mlp,qnn). Default: all found.")
    p_tsn.add_argument("--data-dir", type=str, default="./data/cifar10")
    p_tsn.add_argument("--seed", type=int, default=11224001)
    _add_model_args(p_tsn)

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
        checkpoint_freq=max(0, getattr(args, "checkpoint_freq", 0)),
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
    elif args.stage == "gallery":
        export_gallery(
            output_dir=Path(args.run_dir),
            data_dir=args.data_dir,
            per_class=args.per_class,
        )
    elif args.stage == "probe":
        indices = None
        if args.samples:
            indices = [int(x.strip()) for x in args.samples.split(",")]
        methods = None
        if args.methods:
            methods = [m.strip() for m in args.methods.split(",")]
        probe_checkpoints(
            config=config,
            run_dir=Path(args.run_dir),
            sample_indices=indices,
            data_dir=args.data_dir,
            methods=methods,
        )
    elif args.stage == "tsne":
        methods = None
        if args.methods:
            methods = [m.strip() for m in args.methods.split(",")]
        probe_tsne(
            config=config,
            run_dir=Path(args.run_dir),
            n_per_class=args.n_per_class,
            n_steps=args.n_steps,
            reduction=args.reduction,
            data_dir=args.data_dir,
            methods=methods,
        )
    else:
        raise ValueError(f"Unknown stage: {args.stage!r}")


if __name__ == "__main__":
    main()
