"""Visualize the QCAA HW1 Problem 1 data ranges and target function."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problem1_sample import NUM_SAMPLES, SEED, sample_inputs, target_function


OUTPUT_PATH = Path(__file__).with_name("problem1_data_overview.png")


def main() -> None:
    np.random.seed(SEED)

    train_ranges = np.array([0.0, 0.5] * 2).reshape(2, 2)
    test_ranges = np.array([0.5, 1.0] * 2).reshape(2, 2)

    train_input = sample_inputs(NUM_SAMPLES, train_ranges)
    test_input = sample_inputs(NUM_SAMPLES, test_ranges)
    train_label = target_function(train_input)
    test_label = target_function(test_input)

    x1 = np.linspace(0.0, 1.0, 200)
    x2 = np.linspace(0.0, 1.0, 200)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    full_grid = np.stack([x1_grid.ravel(), x2_grid.ravel()], axis=1)
    full_values = target_function(torch.tensor(full_grid, dtype=torch.float32)).reshape(x1_grid.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    heatmap = axes[0].imshow(
        full_values,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_title("Target Function on [0, 1] x [0, 1]")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].axvline(0.5, color="white", linestyle="--", linewidth=1)
    axes[0].axhline(0.5, color="white", linestyle="--", linewidth=1)
    fig.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04, label="f(x1, x2)")

    scatter_train = axes[1].scatter(
        train_input[:, 0].numpy(),
        train_input[:, 1].numpy(),
        c=train_label.numpy(),
        s=10,
        cmap="viridis",
    )
    axes[1].set_title("Train Samples")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    fig.colorbar(scatter_train, ax=axes[1], fraction=0.046, pad=0.04, label="label")

    scatter_test = axes[2].scatter(
        test_input[:, 0].numpy(),
        test_input[:, 1].numpy(),
        c=test_label.numpy(),
        s=10,
        cmap="viridis",
    )
    axes[2].set_title("Test Samples")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    fig.colorbar(scatter_test, ax=axes[2], fraction=0.046, pad=0.04, label="label")

    fig.suptitle(f"QCAA HW1 Problem 1 overview (seed={SEED})", fontsize=14)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
