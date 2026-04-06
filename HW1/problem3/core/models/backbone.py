"""CNN backbone for feature extraction (outputs 256-D vectors)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNNBackbone(nn.Module):
    """Lightweight CNN for CIFAR-10 → 256-D feature vectors.

    Architecture: 3 conv blocks (32→64→128) with batch norm + max pool,
    followed by adaptive avg pool → flatten → linear → 256-D.
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 32×16×16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32×16×16 → 64×8×8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 64×8×8 → 128×4×4
            nn.Conv2d(128, 128, 3, padding=1) if False else nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Global pool → 128
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))
