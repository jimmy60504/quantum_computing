"""CNN + MLP baseline classifier for CIFAR-10.

Matches the CNN_MLP baseline specified in the homework.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import SimpleCNNBackbone


class CNNMLPClassifier(nn.Module):
    """CNN backbone → single linear head → 10-class output.

    Matches the homework baseline: classifier = nn.Linear(256, 10).
    """

    method_id = "mlp"
    method_label = "CNN + MLP (Baseline)"

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = SimpleCNNBackbone(feature_dim=feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
