"""Shared interfaces for QCAA HW1 Problem 2 methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class MethodSpec:
    id: str
    label: str
    summary: str
    metric_label: str


@dataclass
class BenchmarkResult:
    dataset_name: str
    method_id: str
    method_label: str
    test_accuracy: float | None = None
    trainable_parameters: int | None = None
    kernel_evaluations: int | None = None
    training_time_seconds: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QuantumClassifier(Protocol):
    method_id: str
    method_label: str

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> dict[str, float] | None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        ...

    def count_model_complexity(self) -> int:
        ...
