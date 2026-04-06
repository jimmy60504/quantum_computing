"""CNN + MLP baseline classifier for CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import SimpleCNNBackbone


class CNNMLPClassifier(nn.Module):
    """CNN backbone → MLP head (2 hidden layers) → 10-class output."""

    method_id = "mlp"
    method_label = "CNN + MLP (Baseline)"

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 10,
        hidden_dim: int = 128,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = SimpleCNNBackbone(feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
