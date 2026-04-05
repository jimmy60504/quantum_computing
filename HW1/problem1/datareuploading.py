"""CLI entry point for HW1 Problem 1 data reuploading experiments."""

from __future__ import annotations

import argparse

try:
    from .core.config import (
        Config,
        ENCODING_CHOICES,
        INPUT_ACTIVATION_CHOICES,
        LR_SCHEDULER_CHOICES,
        RENDER_MODE_CHOICES,
        resolve_diff_method,
        validate_device_config,
    )
    from .core.sample import NUM_SAMPLES
    from .core.training import train
except ImportError:  # pragma: no cover - direct script execution on gx10
    from core.config import (
        Config,
        ENCODING_CHOICES,
        INPUT_ACTIVATION_CHOICES,
        LR_SCHEDULER_CHOICES,
        RENDER_MODE_CHOICES,
        resolve_diff_method,
        validate_device_config,
    )
    from core.sample import NUM_SAMPLES
    from core.training import train


def parse_args() -> tuple[Config, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--encoding",
        type=str,
        default="raw",
        choices=ENCODING_CHOICES,
        help="Classical feature lift before the quantum data reuploading circuit.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="inline",
        choices=RENDER_MODE_CHOICES,
        help="Inline viewer rendering or snapshot-only export for remote rendering later.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=LR_SCHEDULER_CHOICES,
        help="Learning-rate schedule applied during training.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Final learning rate used by schedulers that decay over time.",
    )
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument(
        "--input-activation",
        type=str,
        default="tanh",
        choices=INPUT_ACTIVATION_CHOICES,
        help="Nonlinearity applied after the classical projection and before angle scaling.",
    )
    parser.add_argument(
        "--angle-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to projected angles before the quantum circuit.",
    )
    parser.add_argument("--heatmap-grid-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="lightning.qubit",
        choices=("default.qubit", "lightning.qubit"),
    )
    parser.add_argument(
        "--diff-method",
        type=str,
        default=None,
        choices=("backprop", "adjoint"),
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--viewer-export-path", type=str, default=None)
    parser.add_argument("--snapshot-export-path", type=str, default=None)
    parser.add_argument("--viewer-export-every", type=int, default=1)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="hw1-problem1-datareuploading")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    diff_method = resolve_diff_method(args.device, args.diff_method)
    validate_device_config(args.device, diff_method)

    config = Config(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        encoding_mode=args.encoding,
        render_mode=args.render_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        min_learning_rate=args.min_learning_rate,
        hidden_scale=args.hidden_scale,
        input_activation=args.input_activation,
        angle_scale=args.angle_scale,
        heatmap_grid_size=args.heatmap_grid_size,
        device_name=args.device,
        diff_method=diff_method,
        viewer_export_path=args.viewer_export_path,
        snapshot_export_path=args.snapshot_export_path,
        viewer_export_every=args.viewer_export_every,
        render_workers=args.render_workers,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    return config, args.num_samples


def main() -> None:
    config, num_samples = parse_args()
    train(config, num_samples)


if __name__ == "__main__":
    main()
