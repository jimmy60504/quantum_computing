"""Training pipeline for HW1 Problem 1."""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .config import (
        Config,
        resolve_diff_method,
        resolve_runtime_circuit_path,
        resolve_snapshot_export_path,
        resolve_viewer_export_path,
    )
    from .modeling import (
        DataReuploadingRegressor,
        evaluate,
        make_circuit_diagram,
        make_datasets,
        render_timeline_snapshots_parallel,
        serialize_dataset_points,
        snapshot_model_state,
    )
    from .sample import SEED
    from .viewer_io import update_viewer_manifest, write_snapshot_export, write_viewer_export
except ImportError:  # pragma: no cover - direct script execution on gx10
    from config import (
        Config,
        resolve_diff_method,
        resolve_runtime_circuit_path,
        resolve_snapshot_export_path,
        resolve_viewer_export_path,
    )
    from modeling import (
        DataReuploadingRegressor,
        evaluate,
        make_circuit_diagram,
        make_datasets,
        render_timeline_snapshots_parallel,
        serialize_dataset_points,
        snapshot_model_state,
    )
    from sample import SEED
    from viewer_io import update_viewer_manifest, write_snapshot_export, write_viewer_export


def train(config: Config, num_samples: int) -> None:
    torch.manual_seed(SEED)

    train_dataset, test_dataset = make_datasets(num_samples)
    train_points = serialize_dataset_points(train_dataset)
    test_points = serialize_dataset_points(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = (
        DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        if config.render_mode == "inline"
        else None
    )

    model = DataReuploadingRegressor(
        num_qubits=config.num_qubits,
        num_layers=config.num_layers,
        encoding_mode=config.encoding_mode,
        hidden_scale=config.hidden_scale,
        device_name=config.device_name,
        diff_method=config.diff_method or resolve_diff_method(config.device_name, None),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    loss_history: list[dict[str, float]] = []
    timeline_snapshots: list[dict[str, object]] = []
    timeline_steps: list[dict[str, object]] = []

    tracking_uri = config.tracking_uri or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    print("QCAA HW1 Problem 1 - data reuploading baseline")
    print(
        f"seed={SEED} samples={num_samples} qubits={config.num_qubits} "
        f"layers={config.num_layers} encoding={config.encoding_mode} "
        f"batch_size={config.batch_size} "
        f"epochs={config.epochs} lr={config.learning_rate}",
        flush=True,
    )
    print(f"device={config.device_name} diff_method={config.diff_method}", flush=True)
    print(f"trainable_parameters={trainable_parameters}", flush=True)
    print(f"mlflow_tracking_uri={tracking_uri}", flush=True)
    print(f"mlflow_experiment={config.experiment_name}", flush=True)
    print(f"render_mode={config.render_mode}", flush=True)
    print("viewer_steps=batch", flush=True)
    print(f"render_workers={config.render_workers}", flush=True)
    print(flush=True)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(
            {
                "seed": SEED,
                "num_samples": num_samples,
                "num_qubits": config.num_qubits,
                "num_layers": config.num_layers,
                "encoding_mode": config.encoding_mode,
                "render_mode": config.render_mode,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "hidden_scale": config.hidden_scale,
                "heatmap_grid_size": config.heatmap_grid_size,
                "device_name": config.device_name,
                "diff_method": config.diff_method,
                "viewer_export_path": config.viewer_export_path,
                "viewer_export_every": config.viewer_export_every,
                "render_workers": config.render_workers,
                "trainable_parameters": trainable_parameters,
            }
        )

        viewer_export_path = resolve_viewer_export_path(config)
        snapshot_export_path = resolve_snapshot_export_path(config, viewer_export_path)
        runtime_circuit_path = resolve_runtime_circuit_path(viewer_export_path)
        viewer_manifest_path = Path("hf_space_hw1_problem1") / "runtime" / "viewer_manifest.json"

        make_circuit_diagram(model, train_dataset[0][0], runtime_circuit_path)
        if config.render_mode == "inline":
            write_viewer_export(
                config,
                viewer_export_path,
                runtime_circuit_path,
                timeline_steps,
                train_points,
                test_points,
            )
            update_viewer_manifest(
                viewer_manifest_path,
                viewer_export_path,
                config,
                None,
                None,
                None,
                timeline_steps,
            )
        write_snapshot_export(
            config,
            snapshot_export_path,
            viewer_export_path,
            runtime_circuit_path,
            train_points,
            test_points,
            timeline_snapshots,
            loss_history,
            num_samples,
        )

        def format_metric(value: float) -> str:
            return f"{value:.3e}"

        global_step = 0
        for epoch in range(1, config.epochs + 1):
            epoch_snapshots: list[dict[str, object]] = []
            last_batch_index = 0
            last_batch_loss = 0.0
            model.train()
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"epoch {epoch:02d}/{config.epochs:02d}",
                leave=False,
                dynamic_ncols=True,
            )

            for batch_index, (features, labels) in enumerate(progress, start=1):
                optimizer.zero_grad()
                predictions = model(features)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                global_step += 1
                batch_loss = float(loss.item())
                last_batch_index = batch_index
                last_batch_loss = batch_loss
                progress.set_postfix(batch_loss=format_metric(batch_loss), refresh=True)

                if global_step % config.viewer_export_every == 0:
                    epoch_snapshots.append(
                        {
                            "label": f"epoch {epoch:02d} batch {batch_index:02d}",
                            "epoch": epoch,
                            "batch": batch_index,
                            "global_step": global_step,
                            "batch_loss": batch_loss,
                            "model_state": snapshot_model_state(model),
                        }
                    )
                    mlflow.log_metric("batch_loss", batch_loss, step=global_step)

            epoch_summary: dict[str, float] = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "last_batch_loss": last_batch_loss,
            }

            if config.render_mode == "inline":
                train_mse = evaluate(model, train_loader, loss_fn)
                assert test_loader is not None
                test_mse = evaluate(model, test_loader, loss_fn)
                epoch_summary["train_mse"] = train_mse
                epoch_summary["test_mse"] = test_mse
                mlflow.log_metric("train_mse", train_mse, step=epoch)
                mlflow.log_metric("test_mse", test_mse, step=epoch)

            loss_history.append(epoch_summary)
            mlflow.log_metric("epoch_last_batch_loss", last_batch_loss, step=epoch)

            if not epoch_snapshots or epoch_snapshots[-1]["global_step"] != global_step:
                epoch_snapshots.append(
                    {
                        "label": f"epoch {epoch:02d} batch {last_batch_index:02d}",
                        "epoch": epoch,
                        "batch": last_batch_index,
                        "global_step": global_step,
                        "batch_loss": last_batch_loss,
                        "model_state": snapshot_model_state(model),
                    }
                )

            timeline_snapshots.extend(epoch_snapshots)
            write_snapshot_export(
                config,
                snapshot_export_path,
                viewer_export_path,
                runtime_circuit_path,
                train_points,
                test_points,
                timeline_snapshots,
                loss_history,
                num_samples,
            )

            if config.render_mode == "inline":
                rendered_epoch_steps = render_timeline_snapshots_parallel(
                    config,
                    num_samples,
                    epoch_snapshots,
                )
                timeline_steps.extend(rendered_epoch_steps)
                for step in rendered_epoch_steps:
                    mlflow.log_metric(
                        "step_test_mse",
                        float(step["test_mse"]),
                        step=int(step["global_step"]),
                    )

                write_viewer_export(
                    config,
                    viewer_export_path,
                    runtime_circuit_path,
                    timeline_steps,
                    train_points,
                    test_points,
                )
                update_viewer_manifest(
                    viewer_manifest_path,
                    viewer_export_path,
                    config,
                    min(entry["test_mse"] for entry in loss_history if "test_mse" in entry),
                    train_mse,
                    test_mse,
                    timeline_steps,
                )

            if config.render_mode == "inline" and (epoch == 1 or epoch % 5 == 0 or epoch == config.epochs):
                print(
                    f"epoch={epoch:02d} train_mse={format_metric(train_mse)} test_mse={format_metric(test_mse)}",
                    flush=True,
                )
            elif epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
                print(
                    f"epoch={epoch:02d} last_batch_loss={format_metric(last_batch_loss)}",
                    flush=True,
                )

        metric_history = [entry for entry in loss_history if "test_mse" in entry]
        if metric_history:
            best_test_mse = min(entry["test_mse"] for entry in metric_history)
            final_train_mse = metric_history[-1]["train_mse"]
            final_test_mse = metric_history[-1]["test_mse"]
            mlflow.log_metric("best_test_mse", best_test_mse)
            mlflow.log_metric("final_train_mse", final_train_mse)
            mlflow.log_metric("final_test_mse", final_test_mse)
        else:
            best_test_mse = None
            final_train_mse = None
            final_test_mse = None
        make_circuit_diagram(model, train_dataset[0][0], runtime_circuit_path)
        write_snapshot_export(
            config,
            snapshot_export_path,
            viewer_export_path,
            runtime_circuit_path,
            train_points,
            test_points,
            timeline_snapshots,
            loss_history,
            num_samples,
        )
        if config.render_mode == "inline":
            write_viewer_export(
                config,
                viewer_export_path,
                runtime_circuit_path,
                timeline_steps,
                train_points,
                test_points,
            )
            update_viewer_manifest(
                viewer_manifest_path,
                viewer_export_path,
                config,
                best_test_mse,
                final_train_mse,
                final_test_mse,
                timeline_steps,
            )
        mlflow.log_artifact(str(runtime_circuit_path), artifact_path="viewer")
        if config.render_mode == "inline":
            mlflow.log_artifact(str(viewer_export_path), artifact_path="viewer")
            mlflow.log_artifact(str(viewer_manifest_path), artifact_path="viewer")
        mlflow.log_artifact(str(snapshot_export_path), artifact_path="viewer")

        print(flush=True)
        if best_test_mse is not None:
            print(f"best_test_mse={best_test_mse:.6f}", flush=True)
        else:
            print("best_test_mse=deferred_to_snapshot_postprocess", flush=True)
        print(f"runtime_circuit_png={runtime_circuit_path}", flush=True)
        if config.render_mode == "inline":
            print(f"viewer_export={viewer_export_path}", flush=True)
            print(f"viewer_manifest={viewer_manifest_path}", flush=True)
        print(f"snapshot_export={snapshot_export_path}", flush=True)
