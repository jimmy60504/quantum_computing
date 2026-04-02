"""Sample data generation for QCAA HW1 Problem 1."""

import numpy as np
import torch


SEED = 0  # Replace with your student ID if you need assignment-specific randomness.
NUM_SAMPLES = 1000


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.exp(x[:, 0]) + x[:, 1])


def sample_inputs(num_samples: int, ranges: np.ndarray) -> torch.Tensor:
    inputs = torch.zeros(num_samples, 2)

    for i in range(2):
        low, high = ranges[i]
        inputs[:, i] = torch.rand(num_samples) * (high - low) + low

    return inputs


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_ranges = np.array([0.0, 0.5] * 2).reshape(2, 2)
    test_ranges = np.array([0.5, 1.0] * 2).reshape(2, 2)

    train_input = sample_inputs(NUM_SAMPLES, train_ranges)
    test_input = sample_inputs(NUM_SAMPLES, test_ranges)

    train_label = target_function(train_input)
    test_label = target_function(test_input)

    print("QCAA HW1 Problem 1 sample")
    print(f"seed={SEED}")
    print(f"train_input shape={tuple(train_input.shape)}")
    print(f"test_input shape={tuple(test_input.shape)}")
    print(f"train_label shape={tuple(train_label.shape)}")
    print(f"test_label shape={tuple(test_label.shape)}")
    print(f"train x1 range=({train_input[:, 0].min().item():.4f}, {train_input[:, 0].max().item():.4f})")
    print(f"train x2 range=({train_input[:, 1].min().item():.4f}, {train_input[:, 1].max().item():.4f})")
    print(f"test x1 range=({test_input[:, 0].min().item():.4f}, {test_input[:, 0].max().item():.4f})")
    print(f"test x2 range=({test_input[:, 1].min().item():.4f}, {test_input[:, 1].max().item():.4f})")
    print(f"train_label mean={train_label.mean().item():.4f}")
    print(f"test_label mean={test_label.mean().item():.4f}")
    print("first 5 train labels:", train_label[:5].tolist())
    print("first 5 test labels:", test_label[:5].tolist())


if __name__ == "__main__":
    main()
