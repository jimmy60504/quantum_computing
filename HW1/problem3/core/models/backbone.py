"""CNN backbone for feature extraction (outputs 256-D vectors).

Matches the fixed CNNBackbone specified in the homework (DO NOT MODIFY).
Architecture: Conv(3→32) → Conv(32→64) → Conv(64→64) → flatten → 256-D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """Fixed CNN backbone as specified in the homework.

    Input:  3×32×32
    After conv1 + pool: 32×15×15
    After conv2 + pool: 64×6×6
    After conv3 + pool: 64×2×2
    Flatten: 64 * 2 * 2 = 256-D
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=3)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.cnn1(x)))
        x = self.pool(self.relu(self.cnn2(x)))
        x = self.pool(self.relu(self.cnn3(x)))
        x = x.view(x.size(0), -1)  # flatten: 64 * 2 * 2 = 256
        return x
