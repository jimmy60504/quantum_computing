"""Training pipeline for HW1 Problem 2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from ..datasets import DatasetBundle, load_datasets, make_circle_dataset, make_moons_dataset, save_datasets
except ImportError:  # pragma: no cover
    from datasets import DatasetBundle, load_datasets, make_circle_dataset, make_moons_dataset, save_datasets

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


def _build_single_method_models(method_name: str, config: Prob2Config) -> dict[str, object]:
    """Return {dataset_name: model} for a single QML method."""
    if method_name == "explicit":
        return {
            ds: ExplicitQuantumClassifier(
                num_layers=config.num_layers_explicit,
                num_qubits=config.num_qubits,
                learning_rate=config.learning_rate,
                device_name=config.device_name,
                diff_method=config.diff_method,
                seed=config.seed,
            )
            for ds in DATASET_ORDER
        }
    if method_name == "kernel":
        return {
            ds: QuantumKernelClassifier(
                num_qubits=config.num_qubits,
                device_name=config.device_name,
                seed=config.seed,
            )
            for ds in DATASET_ORDER
        }
    if method_name == "reuploading":
        return {
            ds: DataReuploadingClassifier(
                num_layers=config.num_layers_reuploading,
                num_qubits=config.num_qubits,
                learning_rate=config.learning_rate,
                device_name=config.device_name,
                diff_method=config.diff_method,
                seed=config.seed,
            )
            for ds in DATASET_ORDER
        }
    raise ValueError(f"Unknown method: {method_name}")


def _build_timeline_step(
    epoch: int,
    metrics: dict[tuple[str, str], dict[str, float]],
    models: dict[str, dict[str, object]],
    datasets: dict[str, DatasetBundle],
    config: Prob2Config,
    kernel_boundaries: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    top_train_acc = float(np.mean([entry["train_acc"] for entry in metrics.values()]))
    top_test_acc = float(np.mean([entry["test_acc"] for entry in metrics.values()]))
    top_train_loss = float(np.mean([entry["train_loss"] for entry in metrics.values()]))
    top_test_loss = float(np.mean([entry["test_loss"] for entry in metrics.values()]))

    boundaries = {
        method_name: {
            dataset_name: (
                kernel_boundaries[dataset_name]
                if method_name == "kernel" and kernel_boundaries is not None
                else _build_boundary_payload(models[method_name][dataset_name], datasets[dataset_name], config.boundary_grid_size)
            )
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

        _stage("Computing kernel decision boundaries (cached for all export steps)")
        kernel_boundaries: dict[str, dict[str, object]] = {}
        for dataset_name in tqdm(DATASET_ORDER, desc="kernel boundaries", leave=False, dynamic_ncols=True):
            kernel_boundaries[dataset_name] = _build_boundary_payload(
                models["kernel"][dataset_name],
                datasets[dataset_name],
                config.boundary_grid_size,
            )

        _stage("Evaluating initial metrics")
        metrics = {
            (method_name, dataset_name): _collect_metrics(models[method_name][dataset_name], datasets[dataset_name])
            for method_name in METHOD_ORDER
            for dataset_name in DATASET_ORDER
        }
        _log_metrics_to_mlflow(metrics, step=0)

        initial_step = _build_timeline_step(0, metrics, models, datasets, config, kernel_boundaries=kernel_boundaries)
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
                step_payload = _build_timeline_step(epoch, metrics, models, datasets, config, kernel_boundaries=kernel_boundaries)
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


# ---------------------------------------------------------------------------
# Staged execution helpers
# ---------------------------------------------------------------------------


def prepare_datasets_stage(config: Prob2Config, run_dir: Path) -> Path:
    """Generate datasets with fixed seed and save to *run_dir*/datasets.npz."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)

    _stage("Preparing datasets")
    datasets = _load_datasets(config)
    for name, bundle in datasets.items():
        print(
            f"  - {name}: train={bundle.X_train.shape[0]} test={bundle.X_test.shape[0]} "
            f"features={bundle.X_train.shape[1]}",
            flush=True,
        )

    out_path = run_dir / "datasets.npz"
    save_datasets(datasets, out_path)
    print(f"Saved datasets → {out_path}", flush=True)
    return out_path


def train_method_stage(method_name: str, config: Prob2Config, run_dir: Path) -> Path:
    """Train a single QML method, then save a JSON artifact to *run_dir*/{method}_artifact.json.

    The artifact contains epoch-level snapshots (metrics + decision boundary) for each
    dataset, which ``assemble_viewer_stage`` later merges into the viewer JSON.
    """
    run_dir = Path(run_dir)
    datasets_path = run_dir / "datasets.npz"
    if not datasets_path.exists():
        raise FileNotFoundError(f"datasets.npz not found in {run_dir}. Run 'prepare' first.")

    _stage(f"Loading datasets from {datasets_path}")
    datasets = load_datasets(datasets_path)
    for name, bundle in datasets.items():
        print(
            f"  - {name}: train={bundle.X_train.shape[0]} test={bundle.X_test.shape[0]}",
            flush=True,
        )

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    _stage(f"Building {method_name} models")
    models = _build_single_method_models(method_name, config)

    tracking_uri = (
        config.tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    run_display_name = f"{config.run_name}-{method_name}" if config.run_name else method_name
    artifact: dict[str, object] = {
        "method": method_name,
        "config": {
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_qubits": config.num_qubits,
            "num_layers_explicit": config.num_layers_explicit,
            "num_layers_reuploading": config.num_layers_reuploading,
            "boundary_grid_size": config.boundary_grid_size,
            "seed": config.seed,
        },
        "datasets": {ds: {"snapshots": []} for ds in DATASET_ORDER},
    }

    def _snap(ds_name: str, global_step: int, epoch: int, batch: int) -> dict[str, object]:
        m = _collect_metrics(models[ds_name], datasets[ds_name])
        b = _build_boundary_payload(models[ds_name], datasets[ds_name], config.boundary_grid_size)
        label = "Initial" if global_step == 0 else f"E{epoch}·B{batch}"
        return {"global_step": global_step, "epoch": epoch, "batch": batch, "label": label, **m, "boundary": b}

    timings: dict[str, float] = {ds: 0.0 for ds in DATASET_ORDER}

    with mlflow.start_run(run_name=run_display_name):
        mlflow.log_params(
            {
                "method": method_name,
                "seed": config.seed,
                "epochs": config.epochs if method_name != "kernel" else 0,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_qubits": config.num_qubits,
                "num_layers_explicit": config.num_layers_explicit,
                "num_layers_reuploading": config.num_layers_reuploading,
                "boundary_grid_size": config.boundary_grid_size,
            }
        )

        if method_name == "kernel":
            _stage("Fitting quantum kernel")
            for ds_name in tqdm(DATASET_ORDER, desc="kernel fit", leave=False, dynamic_ncols=True):
                start = perf_counter()
                fit_result = models[ds_name].fit(datasets[ds_name].X_train, datasets[ds_name].y_train)
                timings[ds_name] = perf_counter() - start
                print(
                    f"  - {ds_name}: train_acc={fit_result['train_acc']:.3f} "
                    f"evals={models[ds_name].count_model_complexity()}",
                    flush=True,
                )

            _stage("Computing kernel boundaries")
            for ds_name in tqdm(DATASET_ORDER, desc="kernel boundaries", leave=False, dynamic_ncols=True):
                snap = _snap(ds_name, global_step=0, epoch=0, batch=0)
                artifact["datasets"][ds_name]["snapshots"].append(snap)
                mlflow.log_metric(f"{ds_name}/train_acc", snap["train_acc"], step=0)
                mlflow.log_metric(f"{ds_name}/test_acc", snap["test_acc"], step=0)
                mlflow.log_metric(f"{ds_name}/train_loss", snap["train_loss"], step=0)
                mlflow.log_metric(f"{ds_name}/test_loss", snap["test_loss"], step=0)
        else:
            # ── Per-batch training loop ────────────────────────────────────────
            # Build DataLoaders once so shuffle is per-epoch, not per-batch.
            loaders = {
                ds: DataLoader(
                    TensorDataset(
                        torch.as_tensor(datasets[ds].X_train, dtype=torch.float32),
                        torch.as_tensor(datasets[ds].y_train, dtype=torch.float32),
                    ),
                    batch_size=config.batch_size,
                    shuffle=True,
                    drop_last=False,
                )
                for ds in DATASET_ORDER
            }
            n_batches = max(len(loader) for loader in loaders.values())
            total_steps = config.epochs * n_batches

            _stage(f"Computing initial snapshot for {method_name}")
            for ds_name in DATASET_ORDER:
                artifact["datasets"][ds_name]["snapshots"].append(_snap(ds_name, 0, 0, 0))

            _stage(f"Training {method_name}  ({config.epochs} epochs × {n_batches} batches = {total_steps} steps)")
            best_test_acc = max(
                artifact["datasets"][ds]["snapshots"][0]["test_acc"] for ds in DATASET_ORDER
            )
            global_step = 0

            epoch_bar = tqdm(
                range(1, config.epochs + 1),
                total=config.epochs,
                desc=f"{method_name}",
                dynamic_ncols=True,
            )
            for epoch in epoch_bar:
                epoch_iters = {ds: iter(loaders[ds]) for ds in DATASET_ORDER}

                for batch_idx in range(n_batches):
                    global_step += 1
                    is_last_step = (epoch == config.epochs and batch_idx == n_batches - 1)

                    for ds_name in DATASET_ORDER:
                        try:
                            X_batch, y_batch = next(epoch_iters[ds_name])
                        except StopIteration:
                            continue
                        start = perf_counter()
                        models[ds_name].train_step(X_batch.numpy(), y_batch.numpy())
                        timings[ds_name] += perf_counter() - start

                    # Export a snapshot every viewer_export_every steps and at the very end
                    if global_step % config.viewer_export_every == 0 or is_last_step:
                        for ds_name in DATASET_ORDER:
                            snap = _snap(ds_name, global_step, epoch, batch_idx + 1)
                            artifact["datasets"][ds_name]["snapshots"].append(snap)
                            mlflow.log_metric(f"{ds_name}/train_acc", snap["train_acc"], step=global_step)
                            mlflow.log_metric(f"{ds_name}/test_acc", snap["test_acc"], step=global_step)
                            mlflow.log_metric(f"{ds_name}/train_loss", snap["train_loss"], step=global_step)
                            mlflow.log_metric(f"{ds_name}/test_loss", snap["test_loss"], step=global_step)
                            best_test_acc = max(best_test_acc, snap["test_acc"])

                avg_test_acc = float(
                    np.mean([artifact["datasets"][ds]["snapshots"][-1]["test_acc"] for ds in DATASET_ORDER])
                )
                epoch_bar.set_postfix(step=global_step, avg_acc=f"{avg_test_acc:.3f}", best=f"{best_test_acc:.3f}")

            epoch_bar.close()
            mlflow.log_metric("best_test_acc", best_test_acc)

        for ds_name in DATASET_ORDER:
            mlflow.log_metric(f"{ds_name}/training_time_seconds", timings[ds_name])

    artifact_path = run_dir / f"{method_name}_artifact.json"
    with open(artifact_path, "w") as fh:
        json.dump(artifact, fh)
    print(f"Saved artifact → {artifact_path}", flush=True)
    return artifact_path


def assemble_viewer_stage(config: Prob2Config, run_dir: Path) -> tuple[Path, Path]:
    """Read all three method artifacts from *run_dir* and write the viewer JSON.

    Produces a 2-step timeline: initial state (epoch 0) and final state (last recorded epoch).
    """
    run_dir = Path(run_dir)

    _stage(f"Loading datasets from {run_dir / 'datasets.npz'}")
    datasets = load_datasets(run_dir / "datasets.npz")
    scatter = {name: _serialize_scatter_points(bundle) for name, bundle in datasets.items()}

    _stage("Loading method artifacts")
    artifacts: dict[str, dict] = {}
    for method_name in METHOD_ORDER:
        art_path = run_dir / f"{method_name}_artifact.json"
        if not art_path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {art_path}. Run 'train --method {method_name}' first."
            )
        with open(art_path) as fh:
            artifacts[method_name] = json.load(fh)
        n_snaps = sum(
            len(artifacts[method_name]["datasets"][ds]["snapshots"]) for ds in DATASET_ORDER
        )
        print(f"  - {method_name}: {n_snaps} snapshots", flush=True)

    _stage("Building viewer timeline (all exported steps)")

    # Use explicit/circle as the reference timeline — its global_steps drive
    # the viewer slider.  Kernel has only step 0; its boundary is replicated.
    ref_snaps = artifacts["explicit"]["datasets"]["circle"]["snapshots"]
    exported_steps: list[int] = [int(s["global_step"]) for s in ref_snaps]

    # Build lookup: method → dataset → global_step → snapshot
    snap_lookup: dict[str, dict[str, dict[int, dict]]] = {}
    for method_name in METHOD_ORDER:
        snap_lookup[method_name] = {}
        for ds_name in DATASET_ORDER:
            snaps = artifacts[method_name]["datasets"][ds_name]["snapshots"]
            snap_lookup[method_name][ds_name] = {int(s["global_step"]): s for s in snaps}

    timeline_steps: list[dict[str, object]] = []
    for step in exported_steps:
        boundaries: dict[str, dict] = {}
        accuracies: dict[str, dict] = {}
        losses: dict[str, dict] = {}
        all_snaps_at_step: list[dict] = []

        for method_name in METHOD_ORDER:
            boundaries[method_name] = {}
            accuracies[method_name] = {}
            losses[method_name] = {}
            for ds_name in DATASET_ORDER:
                lookup = snap_lookup[method_name][ds_name]
                # Kernel only has step 0; fall back for all other steps
                snap = lookup.get(step) or lookup.get(0) or next(iter(lookup.values()))
                boundaries[method_name][ds_name] = snap["boundary"]
                accuracies[method_name][ds_name] = snap["test_acc"]
                losses[method_name][ds_name] = {
                    "train": snap["train_loss"],
                    "test": snap["test_loss"],
                }
                all_snaps_at_step.append(snap)

        # Recover label / epoch / batch from the reference snapshot
        ref_snap = snap_lookup["explicit"]["circle"].get(step, {})
        label = ref_snap.get("label", "Initial" if step == 0 else f"Step {step}")
        epoch = ref_snap.get("epoch", 0)
        batch = ref_snap.get("batch", 0)

        timeline_steps.append(
            {
                "label": label,
                "epoch": epoch,
                "batch": batch,
                "global_step": step,
                "train_acc": float(np.mean([s["train_acc"] for s in all_snaps_at_step])),
                "test_acc": float(np.mean([s["test_acc"] for s in all_snaps_at_step])),
                "train_loss": float(np.mean([s["train_loss"] for s in all_snaps_at_step])),
                "test_loss": float(np.mean([s["test_loss"] for s in all_snaps_at_step])),
                "accuracies": accuracies,
                "losses": losses,
                "scatter": scatter,
                "boundaries": boundaries,
            }
        )

    best_test_acc = max(
        snap["test_acc"]
        for method_name in METHOD_ORDER
        for ds_name in DATASET_ORDER
        for snap in artifacts[method_name]["datasets"][ds_name]["snapshots"]
    )

    viewer_export_path = resolve_viewer_export_path(config)
    viewer_manifest_path = resolve_viewer_manifest_path(config)
    write_viewer_export(config, viewer_export_path, timeline_steps, scatter, {"best_test_acc": best_test_acc}, use_chunks=True)
    update_viewer_manifest(viewer_manifest_path, viewer_export_path, config, best_test_acc, timeline_steps)

    print(f"Viewer written → {viewer_export_path}", flush=True)
    return viewer_export_path, viewer_manifest_path
