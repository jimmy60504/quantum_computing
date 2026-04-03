"""Benchmark scaffold for QCAA HW1 Problem 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    from .datasets import DatasetBundle, make_circle_dataset, make_moons_dataset
    from .interfaces import BenchmarkResult, MethodSpec
except ImportError:
    from datasets import DatasetBundle, make_circle_dataset, make_moons_dataset
    from interfaces import BenchmarkResult, MethodSpec


DATASET_LOADERS = {
    "circle": make_circle_dataset,
    "moons": make_moons_dataset,
}

METHOD_SPECS = {
    "explicit": MethodSpec(
        id="explicit",
        label="Explicit Quantum Model",
        summary="Trainable variational classifier on top of a fixed encoding circuit.",
        metric_label="trainable_parameters",
    ),
    "kernel": MethodSpec(
        id="kernel",
        label="Implicit Quantum Kernel",
        summary="Fixed quantum feature map plus classical kernel method.",
        metric_label="kernel_evaluations",
    ),
    "reuploading": MethodSpec(
        id="reuploading",
        label="Data Reuploading Circuit",
        summary="Repeated input encoding interleaved with trainable quantum layers.",
        metric_label="trainable_parameters",
    ),
}


@dataclass
class DatasetSummary:
    name: str
    train_size: int
    test_size: int
    class_balance_train: tuple[int, int]
    class_balance_test: tuple[int, int]


def summarize_dataset(bundle: DatasetBundle) -> DatasetSummary:
    train_counts = np.bincount(bundle.y_train, minlength=2)
    test_counts = np.bincount(bundle.y_test, minlength=2)
    return DatasetSummary(
        name=bundle.name,
        train_size=int(bundle.X_train.shape[0]),
        test_size=int(bundle.X_test.shape[0]),
        class_balance_train=(int(train_counts[0]), int(train_counts[1])),
        class_balance_test=(int(test_counts[0]), int(test_counts[1])),
    )


def make_result_skeleton(dataset_name: str, method_id: str) -> BenchmarkResult:
    spec = METHOD_SPECS[method_id]
    return BenchmarkResult(
        dataset_name=dataset_name,
        method_id=spec.id,
        method_label=spec.label,
        notes=[
            "Scaffold only: plug in a concrete model implementation before benchmarking.",
        ],
    )


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "results": root / "results",
        "figures": root / "figures",
        "tables": root / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def timed_call(fn, *args, **kwargs) -> tuple[object, float]:
    start = perf_counter()
    result = fn(*args, **kwargs)
    duration = perf_counter() - start
    return result, duration
