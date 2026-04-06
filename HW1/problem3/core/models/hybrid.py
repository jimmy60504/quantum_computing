"""CNN + Quantum Neural Network hybrid classifier for CIFAR-10."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .backbone import SimpleCNNBackbone


class QuantumHead(nn.Module):
    """Parameterized quantum circuit acting as a classification head.

    Pipeline:
        256-D features → Linear(256, num_qubits) → angle encoding (RY)
        → variational layers (RY + RZ + CNOT ring) × num_layers
        → PauliZ expectations on all qubits
        → Linear(num_qubits, 10) → logits

    This keeps the quantum circuit small (num_qubits wires) while still
    handling the high-dimensional CNN output via classical pre/post layers.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_qubits: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
        device_name: str = "default.qubit",
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        # Classical pre-processing: reduce 256-D → num_qubits
        self.pre = nn.Linear(feature_dim, num_qubits)

        # Variational parameters: (num_layers, num_qubits, 2) for RY + RZ
        self.weights = nn.Parameter(
            0.01 * torch.randn(num_layers, num_qubits, 2, dtype=torch.float32)
        )

        dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Angle encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for layer in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # CNOT ring entanglement
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % num_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self._circuit = torch.vmap(circuit, in_dims=(0, None))
        self._raw_circuit = circuit

        # Classical post-processing: num_qubits expectations → 10 logits
        self.post = nn.Linear(num_qubits, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, feature_dim) → logits: (batch, 10)."""
        x = self.pre(features)  # (batch, num_qubits)
        x = torch.tanh(x) * np.pi  # scale to [-pi, pi] for angle encoding
        expectations = self._circuit(x, self.weights)  # (batch, num_qubits)
        if isinstance(expectations, tuple):
            expectations = torch.stack(expectations, dim=-1)
        return self.post(expectations)


class CNNQNNClassifier(nn.Module):
    """CNN backbone → quantum head → 10-class output."""

    method_id = "qnn"
    method_label = "CNN + QNN (Hybrid)"

    def __init__(
        self,
        feature_dim: int = 256,
        num_qubits: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
        freeze_backbone: bool = False,
        device_name: str = "default.qubit",
        diff_method: str = "backprop",
    ) -> None:
        super().__init__()
        self.backbone = SimpleCNNBackbone(feature_dim=feature_dim)
        self.head = QuantumHead(
            feature_dim=feature_dim,
            num_qubits=num_qubits,
            num_layers=num_layers,
            num_classes=num_classes,
            device_name=device_name,
            diff_method=diff_method,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
