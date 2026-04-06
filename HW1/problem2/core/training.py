"""Training pipeline for HW1 Problem 2."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from ..datasets import DatasetBundle, make_circle_dataset, make_moons_dataset
except ImportError:  # pragma: no cover
    from datasets import DatasetBundle, make_circle_dataset, make_moons_dataset

from .config import Prob2Config, resolve_viewer_export_path, resolve_viewer_manifest_path
from .models import DataReuploadingClassifier, ExplicitQuantumClassifier, QuantumKernelClassifier
from .viewer_io import update_viewer_manifest, write_viewer_export

_EPS = 1e-6
METHOD_ORDER = ("explicit", "kernel", "reuploading")
DATASET_ORDER = ("circle", "moons")


def _stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def _binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    return float(-(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)).mean())


def _serialize_scatter_points(bundle: DatasetBundle) -> list[dict[str, object]]:
    points = getattr(bundle, "X_all_raw", None)
    if points is None:
        points = bundle.X_all
    return [
        {"x": float(features[0]), "y": float(features[1]), "label": int(label)}
        for features, label in zip(points, bundle.y_all, strict=True)
    ]


def _scale_features(features: np.ndarray, bundle: DatasetBundle) -> np.ndarray:
    scaler_mean = getattr(bundle, "scaler_mean", None)
    scaler_scale = getattr(bundle, "scaler_scale", None)
    if scaler_mean is None or scaler_scale is None:
        return features.astype(np.float32)
    return ((features - scaler_mean) / scaler_scale).astype(np.float32)


def _build_boundary_payload(
    model: object,
    bundle: DatasetBundle,
    grid_size: int,
) -> dict[str, object]:
    raw_points = getattr(bundle, "X_all_raw", None)
    if raw_points is None:
        raw_points = bundle.X_all

    x_min, x_max = float(raw_points[:, 0].min() - 0.3), float(raw_points[:, 0].max() + 0.3)
    y_min, y_max = float(raw_points[:, 1].min() - 0.3), float(raw_points[:, 1].max() + 0.3)
    x_axis = np.linspace(x_min, x_max, grid_size)
    y_axis = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_axis, y_axis)

    grid_raw = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    grid_model = _scale_features(grid_raw, bundle)
    z = model.decision_function(grid_model).reshape(len(y_axis), len(x_axis))

    return {
        "x": x_axis.tolist(),
        "y": y_axis.tolist(),
        "z": z.tolist(),
    }


def _collect_metrics(model: object, bundle: DatasetBundle) -> dict[str, float]:
    train_probs = model.decision_function(bundle.X_train)
    test_probs = model.decision_function(bundle.X_test)
    train_pred = (train_probs >= 0.5).astype(np.int64)
    test_pred = (test_probs >= 0.5).astype(np.int64)

    return {
        "train_acc": float((train_pred == bundle.y_train).mean()),
        "test_acc": float((test_pred == bundle.y_test).mean()),
        "train_loss": _binary_cross_entropy(bundle.y_train, train_probs),
        "test_loss": _binary_cross_entropy(bundle.y_test, test_probs),
    }


def _build_models(config: Prob2Config) -> dict[str, dict[str, object]]:
    return {
        "explicit": {
            dataset_name: ExplicitQuantumClassifier(
                num_layers=config.num_layers_explicit,
                num_qubits=config.num_qubits,
                learning_rate=config.learning_rate,
                device_name=config.device_name,
                diff_method=config.diff_method,
                seed=config.seed,
            )
            for dataset_name in DATASET_ORDER
        },
        "kernel": {
            dataset_name: QuantumKernelClassifier(
                num_qubits=config.num_qubits,
                device_name=config.device_name,
                seed=config.seed,
            )
            for dataset_name in DATASET_ORDER
        },
        "reuploading": {
            dataset_name: DataReuploadingClassifier(
                num_layers=config.num_layers_reuploading,
                num_qubits=config.num_qubits,
                learning_rate=config.learning_rate,
                device_name=config.device_name,
                diff_method=config.diff_method,
                seed=config.seed,
            )
            for dataset_name in DATASET_ORDER
        },
    }


def _build_timeline_step(
    epoch: int,
    metrics: dict[tuple[str, str], dict[str, float]],
    models: dict[str, dict[str, object]],
    datasets: dict[str, DatasetBundle],
    config: Prob2Config,
) -> dict[str, object]:
    top_train_acc = float(np.mean([entry["train_acc"] for entry in metrics.values()]))
    top_test_acc = float(np.mean([entry["test_acc"] for entry in metrics.values()]))
    top_train_loss = float(np.mean([entry["train_loss"] for entry in metrics.values()]))
    top_test_loss = float(np.mean([entry["test_loss"] for entry in metrics.values()]))

    boundaries = {
        method_name: {
            dataset_name: _build_boundary_payload(models[method_name][dataset_name], datasets[dataset_name], config.boundary_grid_size)
            for dataset_name in DATASET_ORDER
        }
        for method_name in METHOD_ORDER
    }
    scatter = {dataset_name: _serialize_scatter_points(bundle) for dataset_name, bundle in datasets.items()}
    accuracies = {
        method_name: {
            dataset_name: metrics[(method_name, dataset_name)]["test_acc"]
            for dataset_name in DATASET_ORDER
        }
        for method_name in METHOD_ORDER
    }
    losses = {
        method_name: {
            dataset_name: {
                "train": metrics[(method_name, dataset_name)]["train_loss"],
                "test": metrics[(method_name, dataset_name)]["test_loss"],
            }
            for dataset_name in DATASET_ORDER
        }
        for method_name in METHOD_ORDER
    }

    return {
        "label": "Initial" if epoch == 0 else f"Epoch {epoch}",
        "epoch": epoch,
        "global_step": epoch,
        "train_acc": top_train_acc,
        "test_acc": top_test_acc,
        "train_loss": top_train_loss,
        "test_loss": top_test_loss,
        "accuracies": accuracies,
        "losses": losses,
        "scatter": scatter,
        "boundaries": boundaries,
    }


def _load_datasets(config: Prob2Config) -> dict[str, DatasetBundle]:
    return {
        "circle": make_circle_dataset(n_samples=config.n_samples, random_state=config.seed),
        "moons": make_moons_dataset(n_samples=config.n_samples, random_state=config.seed),
    }


def _log_metrics_to_mlflow(metrics: dict[tuple[str, str], dict[str, float]], step: int) -> None:
    for (method_name, dataset_name), values in metrics.items():
        for metric_name, metric_value in values.items():
            mlflow.log_metric(f"{method_name}/{dataset_name}/{metric_name}", metric_value, step=step)


def train(config: Prob2Config) -> tuple[Path, Path]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    _stage("Preparing datasets")
    datasets = _load_datasets(config)
    for dataset_name, bundle in datasets.items():
        print(
            f"  - {dataset_name}: train={bundle.X_train.shape[0]} test={bundle.X_test.shape[0]} "
            f"features={bundle.X_train.shape[1]}",
            flush=True,
        )

    _stage("Building classifier instances")
    models = _build_models(config)
    print(
        "  - model complexity (initial): "
        f"explicit={models['explicit']['circle'].count_model_complexity()} params, "
        f"reuploading={models['reuploading']['circle'].count_model_complexity()} params, "
        "kernel=precomputed matrix",
        flush=True,
    )

    scatter = {dataset_name: _serialize_scatter_points(bundle) for dataset_name, bundle in datasets.items()}
    timings: dict[tuple[str, str], float] = {(method, dataset): 0.0 for method in METHOD_ORDER for dataset in DATASET_ORDER}
    timeline_steps: list[dict[str, object]] = []

    viewer_export_path = resolve_viewer_export_path(config)
    viewer_manifest_path = resolve_viewer_manifest_path(config)
    tracking_uri = (
        config.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    print("HW1 Problem 2 - QML classifier benchmark", flush=True)
    print(
        f"seed={config.seed} n_samples={config.n_samples} epochs={config.epochs} "
        f"batch_size={config.batch_size} lr={config.learning_rate}",
        flush=True,
    )
    print(
        f"qubits={config.num_qubits} layers(explicit)={config.num_layers_explicit} "
        f"layers(reuploading)={config.num_layers_reuploading}",
        flush=True,
    )
    print(f"device={config.device_name} diff_method={config.diff_method}", flush=True)
    print(f"viewer_export_path={viewer_export_path}", flush=True)
    print(f"mlflow_tracking_uri={tracking_uri}", flush=True)
    print(f"mlflow_experiment={config.experiment_name}", flush=True)

    with mlflow.start_run(run_name=config.run_name or None):
        _stage("Logging configuration to MLflow")
        mlflow.log_params(
            {
                "seed": config.seed,
                "n_samples": config.n_samples,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_qubits": config.num_qubits,
                "num_layers_explicit": config.num_layers_explicit,
                "num_layers_reuploading": config.num_layers_reuploading,
                "boundary_grid_size": config.boundary_grid_size,
                "viewer_export_every": config.viewer_export_every,
                "device_name": config.device_name,
                "diff_method": config.diff_method,
            }
        )

        _stage("Fitting kernel baselines")
        for dataset_name in tqdm(DATASET_ORDER, desc="kernel baselines", leave=False, dynamic_ncols=True):
            start = perf_counter()
            kernel_result = models["kernel"][dataset_name].fit(
                datasets[dataset_name].X_train,
                datasets[dataset_name].y_train,
            )
            timings[("kernel", dataset_name)] += perf_counter() - start
            print(
                f"  - kernel/{dataset_name}: train_acc={kernel_result['train_acc']:.3f} "
                f"evals={models['kernel'][dataset_name].count_model_complexity()}",
                flush=True,
            )

        _stage("Evaluating initial metrics")
        metrics = {
            (method_name, dataset_name): _collect_metrics(models[method_name][dataset_name], datasets[dataset_name])
            for method_name in METHOD_ORDER
            for dataset_name in DATASET_ORDER
        }
        _log_metrics_to_mlflow(metrics, step=0)

        initial_step = _build_timeline_step(0, metrics, models, datasets, config)
        timeline_steps.append(initial_step)
        best_test_acc = max(values["test_acc"] for values in metrics.values())

        _stage("Training explicit and reuploading models")
        epoch_progress = tqdm(
            range(1, config.epochs + 1),
            total=config.epochs,
            desc="training epochs",
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            phase_summaries: list[str] = []
            for method_name in ("explicit", "reuploading"):
                for dataset_name in DATASET_ORDER:
                    epoch_progress.set_postfix_str(f"phase={method_name}/{dataset_name}")
                    start = perf_counter()
                    fit_result = models[method_name][dataset_name].fit(
                        datasets[dataset_name].X_train,
                        datasets[dataset_name].y_train,
                        epochs=1,
                        batch_size=config.batch_size,
                        shuffle=True,
                    )
                    timings[(method_name, dataset_name)] += perf_counter() - start
                    phase_summaries.append(
                        f"{method_name}/{dataset_name}: loss={fit_result['loss']:.3f} "
                        f"train_acc={fit_result['train_acc']:.3f}"
                    )

            metrics = {
                (method_name, dataset_name): _collect_metrics(models[method_name][dataset_name], datasets[dataset_name])
                for method_name in METHOD_ORDER
                for dataset_name in DATASET_ORDER
            }
            _log_metrics_to_mlflow(metrics, step=epoch)
            best_test_acc = max(best_test_acc, max(values["test_acc"] for values in metrics.values()))
            avg_test_acc = float(np.mean([entry["test_acc"] for entry in metrics.values()]))
            epoch_progress.set_postfix(avg_test_acc=f"{avg_test_acc:.3f}", best=f"{best_test_acc:.3f}")

            if epoch % config.viewer_export_every == 0 or epoch == config.epochs:
                _stage(f"Exporting viewer snapshot for epoch {epoch}")
                step_payload = _build_timeline_step(epoch, metrics, models, datasets, config)
                timeline_steps.append(step_payload)

                summary = {
                    "best_test_acc": best_test_acc,
                    "timings": {
                        f"{method_name}/{dataset_name}": timings[(method_name, dataset_name)]
                        for method_name in METHOD_ORDER
                        for dataset_name in DATASET_ORDER
                    },
                    "complexities": {
                        f"{method_name}/{dataset_name}": models[method_name][dataset_name].count_model_complexity()
                        for method_name in METHOD_ORDER
                        for dataset_name in DATASET_ORDER
                    },
                }
                write_viewer_export(config, viewer_export_path, timeline_steps, scatter, summary)
                update_viewer_manifest(viewer_manifest_path, viewer_export_path, config, best_test_acc, timeline_steps)

            if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
                print(
                    f"epoch={epoch:02d} avg_test_acc={avg_test_acc:.3f} "
                    f"best_test_acc={best_test_acc:.3f}",
                    flush=True,
                )
                for phase_summary in phase_summaries:
                    print(f"  - {phase_summary}", flush=True)

        epoch_progress.close()

        _stage("Logging final metrics and artifacts")
        for method_name in METHOD_ORDER:
            for dataset_name in DATASET_ORDER:
                mlflow.log_metric(
                    f"{method_name}/{dataset_name}/complexity",
                    models[method_name][dataset_name].count_model_complexity(),
                )
                mlflow.log_metric(
                    f"{method_name}/{dataset_name}/training_time_seconds",
                    timings[(method_name, dataset_name)],
                )

        mlflow.log_metric("best_test_acc", best_test_acc)
        if viewer_export_path.exists():
            mlflow.log_artifact(str(viewer_export_path), artifact_path="viewer")
        if viewer_manifest_path.exists():
            mlflow.log_artifact(str(viewer_manifest_path), artifact_path="viewer")

    return viewer_export_path, viewer_manifest_path
