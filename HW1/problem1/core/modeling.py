"""Model, dataset, and rendering helpers for HW1 Problem 1."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from .config import Config, TEST_RANGES, TRAIN_RANGES
    from .sample import SEED, sample_inputs, target_function
except ImportError:  # pragma: no cover - direct script execution on gx10
    from config import Config, TEST_RANGES, TRAIN_RANGES
    from sample import SEED, sample_inputs, target_function

_RENDER_CONTEXT: dict[str, object] | None = None


def encoded_feature_dim(encoding_mode: str) -> int:
    if encoding_mode == "raw":
        return 2
    if encoding_mode == "poly":
        return 5
    if encoding_mode == "exp":
        return 2
    raise ValueError(f"Unsupported encoding mode: {encoding_mode}")


def encode_features(x: torch.Tensor, encoding_mode: str) -> torch.Tensor:
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]

    if encoding_mode == "raw":
        return x

    if encoding_mode == "poly":
        return torch.cat([x1, x2, x1.square(), x1 * x2, x2.square()], dim=1)

    if encoding_mode == "exp":
        return torch.cat([torch.exp(x1), x2], dim=1)

    raise ValueError(f"Unsupported encoding mode: {encoding_mode}")


def make_datasets(num_samples: int) -> tuple[TensorDataset, TensorDataset]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_input = sample_inputs(num_samples, TRAIN_RANGES)
    test_input = sample_inputs(num_samples, TEST_RANGES)

    train_label = target_function(train_input).unsqueeze(1)
    test_label = target_function(test_input).unsqueeze(1)

    return TensorDataset(train_input, train_label), TensorDataset(test_input, test_label)


class DataReuploadingRegressor(nn.Module):
    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        encoding_mode: str = "raw",
        hidden_scale: float = 1.0,
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.encoding_mode = encoding_mode
        self.device_name = device_name
        self.diff_method = diff_method

        self.input_projection = nn.Linear(encoded_feature_dim(encoding_mode), num_qubits)
        self.output_head = nn.Linear(num_qubits, 1)
        self.feature_scale = nn.Parameter(torch.full((num_qubits,), hidden_scale))
        self.weights = nn.Parameter(0.05 * torch.randn(num_layers, num_qubits, 3))

        self.dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def circuit(encoded_features: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            for layer in range(num_layers):
                for wire in range(num_qubits):
                    angle = encoded_features[wire]
                    qml.RY(angle, wires=wire)
                    qml.RZ(angle, wires=wire)

                if num_qubits > 1:
                    for wire in range(num_qubits - 1):
                        qml.CNOT(wires=[wire, wire + 1])
                    qml.CNOT(wires=[num_qubits - 1, 0])

                for wire in range(num_qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)

            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lifted_features = encode_features(x, self.encoding_mode)
        encoded = torch.tanh(self.input_projection(lifted_features)) * self.feature_scale
        circuit_outputs = [torch.stack(self.circuit(sample, self.weights)) for sample in encoded]
        q_features = torch.stack(circuit_outputs).to(x.dtype)
        return self.output_head(q_features)


def snapshot_model_state(model: DataReuploadingRegressor) -> dict[str, list]:
    return {
        "input_projection_weight": model.input_projection.weight.detach().cpu().tolist(),
        "input_projection_bias": model.input_projection.bias.detach().cpu().tolist(),
        "output_head_weight": model.output_head.weight.detach().cpu().tolist(),
        "output_head_bias": model.output_head.bias.detach().cpu().tolist(),
        "feature_scale": model.feature_scale.detach().cpu().tolist(),
        "weights": model.weights.detach().cpu().tolist(),
    }


def load_model_state_snapshot(
    model: DataReuploadingRegressor, snapshot_state: dict[str, list]
) -> DataReuploadingRegressor:
    with torch.no_grad():
        model.input_projection.weight.copy_(
            torch.tensor(snapshot_state["input_projection_weight"], dtype=torch.float32)
        )
        model.input_projection.bias.copy_(
            torch.tensor(snapshot_state["input_projection_bias"], dtype=torch.float32)
        )
        model.output_head.weight.copy_(
            torch.tensor(snapshot_state["output_head_weight"], dtype=torch.float32)
        )
        model.output_head.bias.copy_(
            torch.tensor(snapshot_state["output_head_bias"], dtype=torch.float32)
        )
        model.feature_scale.copy_(
            torch.tensor(snapshot_state["feature_scale"], dtype=torch.float32)
        )
        model.weights.copy_(torch.tensor(snapshot_state["weights"], dtype=torch.float32))
    return model


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for features, labels in loader:
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            batch_size = features.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / total_count


def evaluate_tensor(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        return float(nn.functional.mse_loss(predictions, labels).item())


def make_heatmap_cache(
    grid_size: int,
    ranges: np.ndarray,
) -> dict[str, object]:
    x = np.linspace(float(ranges[0, 0]), float(ranges[0, 1]), grid_size, dtype=np.float32)
    y = np.linspace(float(ranges[1, 0]), float(ranges[1, 1]), grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    flat_grid = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    flat_tensor = torch.tensor(flat_grid, dtype=torch.float32)
    target_grid = target_function(flat_tensor).reshape(grid_size, grid_size).numpy()
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "flat_tensor": flat_tensor,
        "target_grid": target_grid,
    }


def build_heatmap_grids_from_cache(
    model: nn.Module,
    heatmap_cache: dict[str, object],
    batch_size: int = 256,
) -> dict[str, dict[str, list[list[float]] | list[float]]]:
    flat_tensor = heatmap_cache["flat_tensor"]
    target_grid = heatmap_cache["target_grid"]
    x = heatmap_cache["x"]
    y = heatmap_cache["y"]
    grid_size = len(x)

    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, flat_tensor.shape[0], batch_size):
            batch = flat_tensor[start : start + batch_size]
            predictions.append(model(batch).squeeze(1).cpu())

    prediction_grid = torch.cat(predictions).reshape(grid_size, grid_size).numpy()
    error_grid = np.abs(target_grid - prediction_grid)

    return {
        "target": {"x": x, "y": y, "z": target_grid.tolist()},
        "prediction": {"x": x, "y": y, "z": prediction_grid.tolist()},
        "error": {"x": x, "y": y, "z": error_grid.tolist()},
    }


def serialize_dataset_points(dataset: TensorDataset) -> dict[str, list[float]]:
    features, labels = dataset.tensors
    return {
        "x1": features[:, 0].detach().cpu().tolist(),
        "x2": features[:, 1].detach().cpu().tolist(),
        "y": labels[:, 0].detach().cpu().tolist(),
    }


def make_circuit_diagram(
    model: DataReuploadingRegressor, sample: torch.Tensor, output_path: Path
) -> Path:
    lifted_sample = encode_features(sample.unsqueeze(0), model.encoding_mode)
    encoded = torch.tanh(model.input_projection(lifted_sample)).squeeze(0) * model.feature_scale
    fig, _ = qml.draw_mpl(model.circuit)(
        encoded.detach().cpu().numpy(),
        model.weights.detach().cpu().numpy(),
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def init_render_worker() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def init_render_worker_context(config_dict: dict[str, object], num_samples: int) -> None:
    global _RENDER_CONTEXT

    init_render_worker()
    train_dataset, test_dataset = make_datasets(num_samples)
    train_features, train_labels = train_dataset.tensors
    test_features, test_labels = test_dataset.tensors
    _RENDER_CONTEXT = {
        "config": config_dict,
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "train_heatmap_cache": make_heatmap_cache(int(config_dict["heatmap_grid_size"]), TRAIN_RANGES),
        "test_heatmap_cache": make_heatmap_cache(int(config_dict["heatmap_grid_size"]), TEST_RANGES),
    }


def resolve_render_start_method() -> str:
    requested = os.environ.get("PROB1_RENDER_MP_START")
    if requested:
        return requested

    if sys.platform.startswith("linux"):
        return "fork"

    return "spawn"


def resolve_render_chunksize(task_count: int, worker_count: int) -> int:
    requested = os.environ.get("PROB1_RENDER_CHUNKSIZE")
    if requested:
        return max(1, int(requested))

    return max(1, task_count // max(1, worker_count * 4))


def _evaluate_timeline_snapshot(
    task: dict[str, object],
    include_heatmaps: bool,
) -> dict[str, object]:
    snapshot = task["snapshot"]
    if _RENDER_CONTEXT is None:
        config_dict = task["config"]
        init_render_worker_context(config_dict, int(config_dict["num_samples"]))
    else:
        config_dict = _RENDER_CONTEXT["config"]

    model = DataReuploadingRegressor(
        num_qubits=int(config_dict["num_qubits"]),
        num_layers=int(config_dict["num_layers"]),
        encoding_mode=str(config_dict["encoding_mode"]),
        hidden_scale=float(config_dict["hidden_scale"]),
        device_name=str(config_dict["device_name"]),
        diff_method=str(config_dict["diff_method"]),
    )
    load_model_state_snapshot(model, snapshot["model_state"])
    train_mse = evaluate_tensor(model, _RENDER_CONTEXT["train_features"], _RENDER_CONTEXT["train_labels"])
    test_mse = evaluate_tensor(model, _RENDER_CONTEXT["test_features"], _RENDER_CONTEXT["test_labels"])
    payload = {
        "label": snapshot["label"],
        "epoch": snapshot["epoch"],
        "batch": snapshot["batch"],
        "global_step": snapshot["global_step"],
        "batch_loss": snapshot["batch_loss"],
        "train_mse": train_mse,
        "test_mse": test_mse,
    }

    if include_heatmaps:
        payload["heatmaps"] = {
            "train": build_heatmap_grids_from_cache(
                model,
                _RENDER_CONTEXT["train_heatmap_cache"],
                batch_size=int(config_dict["batch_size"]),
            ),
            "test": build_heatmap_grids_from_cache(
                model,
                _RENDER_CONTEXT["test_heatmap_cache"],
                batch_size=int(config_dict["batch_size"]),
            ),
        }

    return payload


def evaluate_timeline_snapshot(task: dict[str, object]) -> dict[str, object]:
    return _evaluate_timeline_snapshot(task, include_heatmaps=False)


def render_timeline_snapshot(task: dict[str, object]) -> dict[str, object]:
    return _evaluate_timeline_snapshot(task, include_heatmaps=True)


def _process_timeline_snapshots_parallel(
    config: Config,
    num_samples: int,
    timeline_snapshots: list[dict[str, object]],
    *,
    worker_fn,
    description: str,
) -> list[dict[str, object]]:
    if not timeline_snapshots:
        return []

    config_dict = {
        "num_qubits": config.num_qubits,
        "num_layers": config.num_layers,
        "encoding_mode": config.encoding_mode,
        "hidden_scale": config.hidden_scale,
        "device_name": config.device_name,
        "diff_method": config.diff_method,
        "heatmap_grid_size": config.heatmap_grid_size,
        "batch_size": config.batch_size,
        "num_samples": num_samples,
    }
    tasks = [{"config": config_dict, "snapshot": snapshot} for snapshot in timeline_snapshots]
    worker_count = max(1, min(config.render_workers, len(tasks), os.cpu_count() or 1))

    if worker_count == 1:
        init_render_worker_context(config_dict, num_samples)
        iterator = (worker_fn(task) for task in tasks)
        return list(
            tqdm(
                iterator,
                total=len(tasks),
                desc=description,
                leave=False,
                dynamic_ncols=True,
            )
        )

    ctx = mp.get_context(resolve_render_start_method())
    with ctx.Pool(
        processes=worker_count,
        initializer=init_render_worker_context,
        initargs=(config_dict, num_samples),
    ) as pool:
        iterator = pool.imap(
            worker_fn,
            tasks,
            chunksize=resolve_render_chunksize(len(tasks), worker_count),
        )
        return list(
            tqdm(
                iterator,
                total=len(tasks),
                desc=f"{description} x{worker_count}",
                leave=False,
                dynamic_ncols=True,
            )
        )


def evaluate_timeline_snapshots_parallel(
    config: Config,
    num_samples: int,
    timeline_snapshots: list[dict[str, object]],
) -> list[dict[str, object]]:
    return _process_timeline_snapshots_parallel(
        config,
        num_samples,
        timeline_snapshots,
        worker_fn=evaluate_timeline_snapshot,
        description="evaluate metrics",
    )


def render_timeline_snapshots_parallel(
    config: Config,
    num_samples: int,
    timeline_snapshots: list[dict[str, object]],
) -> list[dict[str, object]]:
    return _process_timeline_snapshots_parallel(
        config,
        num_samples,
        timeline_snapshots,
        worker_fn=render_timeline_snapshot,
        description="render viewer",
    )
