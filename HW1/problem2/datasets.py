"""Dataset helpers for QCAA HW1 Problem 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_all: np.ndarray
    y_all: np.ndarray
    X_train_raw: np.ndarray | None = None
    X_test_raw: np.ndarray | None = None
    X_all_raw: np.ndarray | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None


def _standardize_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 11224001,
) -> DatasetBundle:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_all = scaler.transform(X)

    return DatasetBundle(
        name="",
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        X_all=X_all.astype(np.float32),
        y_all=y.astype(np.int64),
        X_train_raw=X_train.astype(np.float32) * scaler.scale_.astype(np.float32) + scaler.mean_.astype(np.float32),
        X_test_raw=X_test.astype(np.float32) * scaler.scale_.astype(np.float32) + scaler.mean_.astype(np.float32),
        X_all_raw=X.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
    )


def make_circle_dataset(
    n_samples: int = 200,
    noise: float = 0.0,
    test_size: float = 0.3,
    random_state: int = 11224001,
    radius: float = np.sqrt(2 / np.pi),
) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    X = rng.uniform(low=-1.0, high=1.0, size=(n_samples, 2)).astype(np.float32)
    y = (np.linalg.norm(X, axis=1) < radius).astype(np.int64)

    if noise > 0:
        X = X + rng.normal(loc=0.0, scale=noise, size=X.shape).astype(np.float32)

    bundle = _standardize_split(X, y, test_size=test_size, random_state=random_state)
    bundle.name = "circle"
    return bundle


def make_moons_dataset(
    n_samples: int = 200,
    noise: float = 0.1,
    test_size: float = 0.3,
    random_state: int = 11224001,
) -> DatasetBundle:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    bundle = _standardize_split(
        X.astype(np.float32),
        y.astype(np.int64),
        test_size=test_size,
        random_state=random_state,
    )
    bundle.name = "moons"
    return bundle


def save_datasets(bundles: dict[str, DatasetBundle], path: str | Path) -> None:
    """Persist a dict of DatasetBundle objects to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, bundle in bundles.items():
        arrays[f"{name}__X_train"] = bundle.X_train
        arrays[f"{name}__X_test"] = bundle.X_test
        arrays[f"{name}__y_train"] = bundle.y_train
        arrays[f"{name}__y_test"] = bundle.y_test
        arrays[f"{name}__X_all"] = bundle.X_all
        arrays[f"{name}__y_all"] = bundle.y_all
        for field in ("X_train_raw", "X_test_raw", "X_all_raw", "scaler_mean", "scaler_scale"):
            val = getattr(bundle, field, None)
            if val is not None:
                arrays[f"{name}__{field}"] = val
    np.savez(str(path), **arrays)


def load_datasets(path: str | Path) -> dict[str, DatasetBundle]:
    """Load a dict of DatasetBundle objects from a .npz file."""
    path = Path(path)
    data = np.load(str(path))
    files = set(data.files)
    names = sorted({k.split("__")[0] for k in files if k.endswith("__X_train")})
    result: dict[str, DatasetBundle] = {}
    for name in names:
        def _opt(field: str, _n: str = name) -> np.ndarray | None:
            key = f"{_n}__{field}"
            return data[key] if key in files else None

        result[name] = DatasetBundle(
            name=name,
            X_train=data[f"{name}__X_train"],
            X_test=data[f"{name}__X_test"],
            y_train=data[f"{name}__y_train"],
            y_test=data[f"{name}__y_test"],
            X_all=data[f"{name}__X_all"],
            y_all=data[f"{name}__y_all"],
            X_train_raw=_opt("X_train_raw"),
            X_test_raw=_opt("X_test_raw"),
            X_all_raw=_opt("X_all_raw"),
            scaler_mean=_opt("scaler_mean"),
            scaler_scale=_opt("scaler_scale"),
        )
    return result
