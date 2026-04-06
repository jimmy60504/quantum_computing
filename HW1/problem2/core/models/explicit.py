"""Explicit variational quantum classifier for HW1 Problem 2."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_EPS = 1e-6


class ExplicitQuantumClassifier(nn.Module):
    method_id = "explicit"
    method_label = "Explicit Quantum Model"

    def __init__(
        self,
        num_layers: int = 4,
        num_qubits: int = 2,
        learning_rate: float = 0.05,
        device_name: str = "default.qubit",
        diff_method: str = "backprop",
        seed: int = 11224001,
    ) -> None:
        super().__init__()
        if num_qubits != 2:
            raise ValueError("ExplicitQuantumClassifier currently expects num_qubits=2.")

        torch.manual_seed(seed)
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.device_name = device_name
        self.diff_method = diff_method
        self.weights = nn.Parameter(0.01 * torch.randn(num_layers, num_qubits, 2, dtype=torch.float32))

        self.dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def circuit(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            qml.RX(features[0], wires=0)
            qml.RX(features[1], wires=1)
            for layer in range(num_layers):
                for wire in range(num_qubits):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        self._raw_circuit = circuit
        self._circuit = torch.vmap(circuit, in_dims=(0, None))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCELoss()

    def _as_tensor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            tensor = X.detach().clone().to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def forward(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        features = self._as_tensor(X)
        expvals = self._circuit(features, self.weights).reshape(-1)
        return torch.clamp((expvals + 1.0) / 2.0, _EPS, 1.0 - _EPS)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> dict[str, float]:
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        last_loss = 0.0
        last_acc = 0.0
        for _ in range(epochs):
            self.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for features, labels in loader:
                self.optimizer.zero_grad()
                probabilities = self.forward(features)
                labels = labels.to(probabilities.dtype)
                loss = self.loss_fn(probabilities, labels)
                loss.backward()
                self.optimizer.step()

                batch_size_actual = int(labels.shape[0])
                total_loss += float(loss.item()) * batch_size_actual
                total_correct += int(((probabilities >= 0.5).to(torch.int64) == labels.to(torch.int64)).sum().item())
                total_seen += batch_size_actual

            last_loss = total_loss / max(total_seen, 1)
            last_acc = total_correct / max(total_seen, 1)

        return {"loss": last_loss, "train_acc": last_acc}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.decision_function(X)
        return (probabilities >= 0.5).astype(np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(X)
        return probabilities.detach().cpu().numpy().astype(np.float64)

    def count_model_complexity(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))
