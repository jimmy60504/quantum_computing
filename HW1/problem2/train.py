"""CLI entry point for HW1 Problem 2 training."""

from __future__ import annotations

import argparse

try:
    from .core.config import Prob2Config
    from .core.training import train
except ImportError:  # pragma: no cover - direct script execution on gx10
    from core.config import Prob2Config
    from core.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train HW1 Problem 2 QML classifiers.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--layers-explicit", type=int, default=4)
    parser.add_argument("--layers-reuploading", type=int, default=4)
    parser.add_argument("--boundary-grid-size", type=int, default=50)
    parser.add_argument("--viewer-export-every", type=int, default=5)
    parser.add_argument("--viewer-export-path", type=str, default="HW1/problem2/hf_space/runtime")
    parser.add_argument("--tracking-uri", type=str, default="")
    parser.add_argument("--experiment-name", type=str, default="hw1-problem2-qml-classifiers")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", type=str, default="default.qubit")
    parser.add_argument("--diff-method", type=str, default="backprop")
    return parser


def parse_args() -> Prob2Config:
    args = build_parser().parse_args()
    return Prob2Config(
        n_samples=args.n_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_qubits=args.num_qubits,
        num_layers_explicit=args.layers_explicit,
        num_layers_reuploading=args.layers_reuploading,
        boundary_grid_size=args.boundary_grid_size,
        viewer_export_every=max(1, args.viewer_export_every),
        viewer_export_path=args.viewer_export_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        device_name=args.device,
        diff_method=args.diff_method,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
