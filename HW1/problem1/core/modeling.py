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
    if encoding_mode in (
        "quantum_exact",
        "phase_learnable",
        "scaled_exact",
        "same_axis_reupload",
        "same_axis_rot",
    ):
        return 2
    raise ValueError(f"Unsupported encoding mode: {encoding_mode}")


def encode_features(x: torch.Tensor, encoding_mode: str) -> torch.Tensor:
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]

    if encoding_mode in (
        "quantum_exact",
        "phase_learnable",
        "scaled_exact",
        "same_axis_reupload",
        "same_axis_rot",
    ):
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
        encoding_mode: str = "phase_learnable",
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.encoding_mode = encoding_mode
        self.use_quantum_exact = encoding_mode == "quantum_exact"
        self.use_phase_learnable = encoding_mode == "phase_learnable"
        self.use_scaled_exact = encoding_mode == "scaled_exact"
        self.use_same_axis_reupload = encoding_mode == "same_axis_reupload"
        self.use_same_axis_rot = encoding_mode == "same_axis_rot"
        self.device_name = device_name
        self.diff_method = diff_method

        if encoding_mode not in (
            "quantum_exact",
            "phase_learnable",
            "scaled_exact",
            "same_axis_reupload",
            "same_axis_rot",
        ):
            raise ValueError(f"Unsupported encoding mode: {encoding_mode}")
        if num_qubits != 1:
            raise ValueError(f"{encoding_mode} mode requires num_qubits=1.")
        if not (self.use_same_axis_reupload or self.use_same_axis_rot) and num_layers != 1:
            raise ValueError(f"{encoding_mode} mode requires num_layers=1.")
        if (self.use_same_axis_reupload or self.use_same_axis_rot) and num_layers < 1:
            raise ValueError(f"{encoding_mode} requires num_layers >= 1.")

        self.phase_shift = (
            nn.Parameter(torch.tensor(-np.pi / 2.0, dtype=torch.float32))
            if self.use_phase_learnable or self.use_scaled_exact
            else None
        )
        self.exp_scale = (
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            if self.use_scaled_exact
            else None
        )
        self.exp_bias = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            if self.use_scaled_exact
            else None
        )
        self.x2_scale = (
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            if self.use_scaled_exact
            else None
        )
        self.x2_bias = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            if self.use_scaled_exact
            else None
        )
        self.phase_shifts = (
            nn.Parameter(torch.full((num_layers,), -np.pi / 2.0, dtype=torch.float32))
            if self.use_same_axis_reupload or self.use_same_axis_rot
            else None
        )
        self.exp_scales = (
            nn.Parameter(torch.ones(num_layers, dtype=torch.float32))
            if self.use_same_axis_reupload or self.use_same_axis_rot
            else None
        )
        self.exp_biases = (
            nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
            if self.use_same_axis_reupload or self.use_same_axis_rot
            else None
        )
        self.x2_scales = (
            nn.Parameter(torch.ones(num_layers, dtype=torch.float32))
            if self.use_same_axis_reupload or self.use_same_axis_rot
            else None
        )
        self.x2_biases = (
            nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
            if self.use_same_axis_reupload or self.use_same_axis_rot
            else None
        )
        self.block_rotations = (
            nn.Parameter(torch.zeros((num_layers, 3), dtype=torch.float32))
            if self.use_same_axis_rot
            else None
        )

        self.dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def exact_circuit(
            encoded_features: torch.Tensor,
            phase_shifts: torch.Tensor,
            exp_scales: torch.Tensor,
            exp_biases: torch.Tensor,
            x2_scales: torch.Tensor,
            x2_biases: torch.Tensor,
            block_rotations: torch.Tensor,
        ) -> torch.Tensor:
            for block_idx in range(self.num_layers):
                qml.RY(exp_scales[block_idx] * encoded_features[0] + exp_biases[block_idx], wires=0)
                qml.RY(x2_scales[block_idx] * encoded_features[1] + x2_biases[block_idx], wires=0)
                qml.RY(phase_shifts[block_idx], wires=0)
                qml.Rot(
                    block_rotations[block_idx, 0],
                    block_rotations[block_idx, 1],
                    block_rotations[block_idx, 2],
                    wires=0,
                )
            return qml.expval(qml.PauliZ(0))

        self.exact_circuit = exact_circuit

    def _resolve_circuit_parameters(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_same_axis_reupload or self.use_same_axis_rot:
            block_rotations = (
                self.block_rotations.to(dtype=x.dtype, device=x.device)
                if self.block_rotations is not None
                else torch.zeros((self.num_layers, 3), dtype=x.dtype, device=x.device)
            )
            return (
                self.phase_shifts.to(dtype=x.dtype, device=x.device),
                self.exp_scales.to(dtype=x.dtype, device=x.device),
                self.exp_biases.to(dtype=x.dtype, device=x.device),
                self.x2_scales.to(dtype=x.dtype, device=x.device),
                self.x2_biases.to(dtype=x.dtype, device=x.device),
                block_rotations,
            )

        phase_shift = (
            self.phase_shift
            if self.phase_shift is not None
            else torch.tensor(-np.pi / 2.0, dtype=x.dtype, device=x.device)
        )
        exp_scale = (
            self.exp_scale
            if self.exp_scale is not None
            else torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
        exp_bias = (
            self.exp_bias
            if self.exp_bias is not None
            else torch.tensor(0.0, dtype=x.dtype, device=x.device)
        )
        x2_scale = (
            self.x2_scale
            if self.x2_scale is not None
            else torch.tensor(1.0, dtype=x.dtype, device=x.device)
        )
        x2_bias = (
            self.x2_bias
            if self.x2_bias is not None
            else torch.tensor(0.0, dtype=x.dtype, device=x.device)
        )
        return (
            torch.atleast_1d(phase_shift).to(dtype=x.dtype, device=x.device),
            torch.atleast_1d(exp_scale).to(dtype=x.dtype, device=x.device),
            torch.atleast_1d(exp_bias).to(dtype=x.dtype, device=x.device),
            torch.atleast_1d(x2_scale).to(dtype=x.dtype, device=x.device),
            torch.atleast_1d(x2_bias).to(dtype=x.dtype, device=x.device),
            torch.zeros((self.num_layers, 3), dtype=x.dtype, device=x.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lifted_features = encode_features(x, self.encoding_mode)
        phase_shifts, exp_scales, exp_biases, x2_scales, x2_biases, block_rotations = (
            self._resolve_circuit_parameters(x)
        )
        circuit_outputs = [
            self.exact_circuit(
                sample,
                phase_shifts,
                exp_scales,
                exp_biases,
                x2_scales,
                x2_biases,
                block_rotations,
            )
            for sample in lifted_features
        ]
        return torch.stack(circuit_outputs).unsqueeze(1).to(x.dtype)


def snapshot_model_state(model: DataReuploadingRegressor) -> dict[str, list]:
    snapshot: dict[str, float] = {}
    if model.phase_shifts is not None:
        snapshot["phase_shifts"] = model.phase_shifts.detach().cpu().tolist()
    if model.block_rotations is not None:
        snapshot["block_rotations"] = model.block_rotations.detach().cpu().tolist()
    if model.exp_scales is not None:
        snapshot["exp_scales"] = model.exp_scales.detach().cpu().tolist()
    if model.exp_biases is not None:
        snapshot["exp_biases"] = model.exp_biases.detach().cpu().tolist()
    if model.x2_scales is not None:
        snapshot["x2_scales"] = model.x2_scales.detach().cpu().tolist()
    if model.x2_biases is not None:
        snapshot["x2_biases"] = model.x2_biases.detach().cpu().tolist()
    if model.phase_shift is not None:
        snapshot["phase_shift"] = float(model.phase_shift.detach().cpu().item())
    if model.exp_scale is not None:
        snapshot["exp_scale"] = float(model.exp_scale.detach().cpu().item())
    if model.exp_bias is not None:
        snapshot["exp_bias"] = float(model.exp_bias.detach().cpu().item())
    if model.x2_scale is not None:
        snapshot["x2_scale"] = float(model.x2_scale.detach().cpu().item())
    if model.x2_bias is not None:
        snapshot["x2_bias"] = float(model.x2_bias.detach().cpu().item())
    return snapshot


def load_model_state_snapshot(
    model: DataReuploadingRegressor, snapshot_state: dict[str, list]
) -> DataReuploadingRegressor:
    with torch.no_grad():
        if model.phase_shifts is not None:
            model.phase_shifts.copy_(torch.tensor(snapshot_state["phase_shifts"], dtype=torch.float32))
        if model.block_rotations is not None:
            model.block_rotations.copy_(
                torch.tensor(snapshot_state["block_rotations"], dtype=torch.float32)
            )
        if model.exp_scales is not None:
            model.exp_scales.copy_(torch.tensor(snapshot_state["exp_scales"], dtype=torch.float32))
        if model.exp_biases is not None:
            model.exp_biases.copy_(torch.tensor(snapshot_state["exp_biases"], dtype=torch.float32))
        if model.x2_scales is not None:
            model.x2_scales.copy_(torch.tensor(snapshot_state["x2_scales"], dtype=torch.float32))
        if model.x2_biases is not None:
            model.x2_biases.copy_(torch.tensor(snapshot_state["x2_biases"], dtype=torch.float32))
        if model.phase_shift is not None:
            model.phase_shift.copy_(torch.tensor(snapshot_state["phase_shift"], dtype=torch.float32))
        if model.exp_scale is not None:
            model.exp_scale.copy_(torch.tensor(snapshot_state["exp_scale"], dtype=torch.float32))
        if model.exp_bias is not None:
            model.exp_bias.copy_(torch.tensor(snapshot_state["exp_bias"], dtype=torch.float32))
        if model.x2_scale is not None:
            model.x2_scale.copy_(torch.tensor(snapshot_state["x2_scale"], dtype=torch.float32))
        if model.x2_bias is not None:
            model.x2_bias.copy_(torch.tensor(snapshot_state["x2_bias"], dtype=torch.float32))
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
    structured_sample = lifted_sample.squeeze(0)
    phase_shifts, exp_scales, exp_biases, x2_scales, x2_biases, block_rotations = (
        model._resolve_circuit_parameters(structured_sample)
    )
    fig, _ = qml.draw_mpl(model.exact_circuit)(
        structured_sample.detach().cpu().numpy(),
        phase_shifts.detach().cpu().numpy(),
        exp_scales.detach().cpu().numpy(),
        exp_biases.detach().cpu().numpy(),
        x2_scales.detach().cpu().numpy(),
        x2_biases.detach().cpu().numpy(),
        block_rotations.detach().cpu().numpy(),
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
