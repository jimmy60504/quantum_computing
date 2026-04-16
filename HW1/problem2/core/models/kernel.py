"""Implicit quantum kernel classifier for HW1 Problem 2."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.svm import SVC


class QuantumKernelClassifier:
    method_id = "kernel"
    method_label = "Implicit Quantum Kernel"

    def __init__(
        self,
        num_qubits: int = 2,
        device_name: str = "default.qubit",
        seed: int = 11224001,
    ) -> None:
        self.num_qubits = num_qubits
        self.device_name = device_name
        self.seed = seed
        self.dev = qml.device(device_name, wires=num_qubits)
        self.model = SVC(kernel="precomputed", probability=True, random_state=seed)
        self._X_train: np.ndarray | None = None
        self._kernel_evaluations = 0

        @qml.qnode(self.dev)
        def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)
            return qml.probs(wires=range(num_qubits))

        self._kernel_circuit = kernel_circuit

    def _feature_map(self, x: np.ndarray) -> None:
        for i in range(self.num_qubits):
            qml.Hadamard(wires=i)
        for i in range(self.num_qubits):
            qml.RZ(x[i], wires=i)
        for i in range(self.num_qubits):
            j = (i + 1) % self.num_qubits
            qml.CNOT(wires=[i, j])
            qml.RZ(x[i] * x[j], wires=j)
            qml.CNOT(wires=[i, j])

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        probabilities = self._kernel_circuit(x1, x2)
        return float(np.real(probabilities[0]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        self._X_train = np.asarray(X, dtype=np.float64)
        y_train = np.asarray(y, dtype=np.int64)
        train_kernel = qml.kernels.kernel_matrix(self._X_train, self._X_train, self.kernel)
        self._kernel_evaluations = int(train_kernel.size)
        self.model.fit(train_kernel, y_train)
        train_predictions = self.model.predict(train_kernel)
        train_accuracy = float((train_predictions == y_train).mean())
        return {"train_acc": train_accuracy, "loss": 0.0}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.decision_function(X)
        return (probabilities >= 0.5).astype(np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("Call fit() before decision_function().")

        features = np.asarray(X, dtype=np.float64)
        # Evaluate kernel only against support vectors to avoid O(n_train) evals per test point.
        support_vectors = self._X_train[self.model.support_]
        eval_kernel = qml.kernels.kernel_matrix(features, support_vectors, self.kernel)
        decision_values = (eval_kernel @ self.model.dual_coef_.T).ravel() + self.model.intercept_[0]
        probabilities = 1.0 / (1.0 + np.exp(-decision_values))
        return np.asarray(probabilities, dtype=np.float64)

    def count_model_complexity(self) -> int:
        return int(self._kernel_evaluations)
