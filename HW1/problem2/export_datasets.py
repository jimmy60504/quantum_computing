"""Export synthetic datasets for QCAA HW1 Problem 2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .benchmark import DATASET_LOADERS, ensure_output_dirs
except ImportError:
    from benchmark import DATASET_LOADERS, ensure_output_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Problem 2 datasets to disk.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("HW1") / "problem2",
        help="Problem 2 root directory.",
    )
    return parser


def write_csv(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x1", "x2", "label"])
        for features, label in zip(X, y, strict=True):
            writer.writerow([float(features[0]), float(features[1]), int(label)])


def export_bundle(name: str, root: Path) -> None:
    bundle = DATASET_LOADERS[name]()
    data_dir = root / "data"
    fig_dir = root / "figures"

    npz_path = data_dir / f"{name}_dataset.npz"
    np.savez(
        npz_path,
        X_train=bundle.X_train,
        X_test=bundle.X_test,
        y_train=bundle.y_train,
        y_test=bundle.y_test,
        X_all=bundle.X_all,
        y_all=bundle.y_all,
    )

    write_csv(data_dir / f"{name}_train.csv", bundle.X_train, bundle.y_train)
    write_csv(data_dir / f"{name}_test.csv", bundle.X_test, bundle.y_test)
    write_csv(data_dir / f"{name}_all.csv", bundle.X_all, bundle.y_all)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, X, y, title in [
        (axes[0], bundle.X_train, bundle.y_train, "Train"),
        (axes[1], bundle.X_test, bundle.y_test, "Test"),
    ]:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, alpha=0.85)
        ax.set_title(f"{name} {title}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.savefig(fig_dir / f"{name}_preview.png", dpi=160)
    plt.close(fig)

    print(f"exported {name}:")
    print(f"  npz: {npz_path}")
    print(f"  csv: {data_dir / f'{name}_train.csv'}")
    print(f"  fig: {fig_dir / f'{name}_preview.png'}")


def main() -> None:
    args = build_parser().parse_args()
    paths = ensure_output_dirs(args.output_dir)
    for name in DATASET_LOADERS:
        export_bundle(name, paths["root"])


if __name__ == "__main__":
    main()
