"""First-pass PennyLane data reuploading baseline for QCAA HW1 Problem 1."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from problem1_sample import NUM_SAMPLES, SEED, sample_inputs, target_function


@dataclass
class Config:
    num_qubits: int = 2
    num_layers: int = 3
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.03
    hidden_scale: float = 1.0
    heatmap_grid_size: int = 64
    device_name: str = "lightning.qubit"
    diff_method: str | None = None
    viewer_export_path: str | None = None
    viewer_export_every: int = 1
    tracking_uri: str | None = None
    experiment_name: str = "hw1-problem1-datareuploading"
    run_name: str | None = None


def make_datasets(num_samples: int) -> tuple[TensorDataset, TensorDataset]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_ranges = np.array([0.0, 0.5] * 2).reshape(2, 2)
    test_ranges = np.array([0.5, 1.0] * 2).reshape(2, 2)

    train_input = sample_inputs(num_samples, train_ranges)
    test_input = sample_inputs(num_samples, test_ranges)

    train_label = target_function(train_input).unsqueeze(1)
    test_label = target_function(test_input).unsqueeze(1)

    return TensorDataset(train_input, train_label), TensorDataset(test_input, test_label)


class DataReuploadingRegressor(nn.Module):
    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        hidden_scale: float = 1.0,
        device_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device_name = device_name
        self.diff_method = diff_method

        self.input_projection = nn.Linear(2, num_qubits)
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
        encoded = torch.tanh(self.input_projection(x)) * self.feature_scale
        circuit_outputs = [torch.stack(self.circuit(sample, self.weights)) for sample in encoded]
        q_features = torch.stack(circuit_outputs).to(x.dtype)
        return self.output_head(q_features)


def resolve_diff_method(device_name: str, requested_diff_method: str | None) -> str:
    if requested_diff_method is not None:
        return requested_diff_method

    if device_name == "default.qubit":
        return "backprop"

    if device_name == "lightning.qubit":
        return "adjoint"

    raise ValueError(f"Unsupported device: {device_name}")


def validate_device_config(device_name: str, diff_method: str) -> None:
    supported_pairs = {
        ("default.qubit", "backprop"),
        ("default.qubit", "adjoint"),
        ("lightning.qubit", "adjoint"),
    }
    if (device_name, diff_method) not in supported_pairs:
        supported_text = ", ".join(
            f"{device} + {method}" for device, method in sorted(supported_pairs)
        )
        raise ValueError(
            f"Unsupported device/diff combination: {device_name} + {diff_method}. "
            f"Try one of: {supported_text}."
        )


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "run"


def make_default_export_stem(config: Config) -> str:
    if config.run_name:
        return slugify(config.run_name)

    return slugify(
        (
            f"q{config.num_qubits}-l{config.num_layers}-"
            f"{config.device_name}-{config.diff_method}-"
            f"lr{config.learning_rate}-b{config.batch_size}-e{config.epochs}"
        )
    )


def resolve_viewer_export_path(config: Config) -> Path:
    if config.viewer_export_path:
        return Path(config.viewer_export_path)
    return Path("hf_space_hw1") / "runtime" / f"{make_default_export_stem(config)}.json"

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


def build_heatmap_grids(
    model: nn.Module, grid_size: int, batch_size: int = 256
) -> dict[str, dict[str, list[list[float]] | list[float]]]:
    x = np.linspace(0.5, 1.0, grid_size, dtype=np.float32)
    y = np.linspace(0.5, 1.0, grid_size, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    flat_grid = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    flat_tensor = torch.tensor(flat_grid, dtype=torch.float32)

    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, flat_tensor.shape[0], batch_size):
            batch = flat_tensor[start : start + batch_size]
            predictions.append(model(batch).squeeze(1).cpu())

    prediction_grid = torch.cat(predictions).reshape(grid_size, grid_size).numpy()
    target_grid = target_function(flat_tensor).reshape(grid_size, grid_size).numpy()
    error_grid = np.abs(target_grid - prediction_grid)

    return {
        "target": {"x": x.tolist(), "y": y.tolist(), "z": target_grid.tolist()},
        "prediction": {"x": x.tolist(), "y": y.tolist(), "z": prediction_grid.tolist()},
        "error": {"x": x.tolist(), "y": y.tolist(), "z": error_grid.tolist()},
    }


def make_loss_curve(loss_history: list[dict[str, float]], output_path: Path) -> Path:
    epochs = [entry["epoch"] for entry in loss_history]
    train_mse = [entry["train_mse"] for entry in loss_history]
    test_mse = [entry["test_mse"] for entry in loss_history]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, train_mse, marker="o", label="train MSE")
    ax.plot(epochs, test_mse, marker="s", label="test MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("HW1 Problem 1 data reuploading baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_circuit_diagram(
    model: DataReuploadingRegressor, sample: torch.Tensor, output_path: Path
) -> Path:
    encoded = torch.tanh(model.input_projection(sample.unsqueeze(0))).squeeze(0) * model.feature_scale
    fig, _ = qml.draw_mpl(model.circuit)(
        encoded.detach().cpu().numpy(),
        model.weights.detach().cpu().numpy(),
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_prediction_heatmap(
    heatmap_grids: dict[str, dict[str, list[list[float]] | list[float]]], output_path: Path
) -> Path:
    target_grid = np.asarray(heatmap_grids["target"]["z"], dtype=np.float32)
    prediction_grid = np.asarray(heatmap_grids["prediction"]["z"], dtype=np.float32)
    error_grid = np.asarray(heatmap_grids["error"]["z"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    images = []
    titles = ["Target on test domain", "Model prediction", "Absolute error"]
    values = [target_grid, prediction_grid, error_grid]
    cmaps = ["viridis", "viridis", "magma"]

    for ax, title, value, cmap in zip(axes, titles, values, cmaps):
        image = ax.imshow(
            value,
            extent=(0.5, 1.0, 0.5, 1.0),
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        images.append(image)

    fig.colorbar(images[0], ax=axes[0], fraction=0.046, pad=0.04, label="target")
    fig.colorbar(images[1], ax=axes[1], fraction=0.046, pad=0.04, label="prediction")
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.04, label="abs error")
    fig.suptitle("HW1 Problem 1 prediction heatmap on test domain", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_viewer_export(
    config: Config,
    viewer_export_path: Path,
    timeline_steps: list[dict[str, object]],
) -> Path:
    viewer_export_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": "QCAA HW1 Problem 1",
        "subtitle": "Data reuploading regression viewer",
        "status": "trajectory export",
        "description": (
            "This export is generated from the latest training run and stores raw heatmap "
            "grids per batch step for Plotly playback."
        ),
        "experiment": {
            "model": "PennyLane data reuploading regressor",
            "task": "Regression on f(x1, x2) = sin(exp(x1) + x2)",
            "train_domain": "[0.0, 0.5] x [0.0, 0.5]",
            "test_domain": "[0.5, 1.0] x [0.5, 1.0]",
            "device": f"{config.device_name} + {config.diff_method}",
            "note": "Slider steps correspond to exported training batches."
        },
        "assets": {
            "circuit": "runtime/problem1_circuit.png",
            "data_overview": "assets/problem1_data_overview.png"
        },
        "grid": {
            "x_min": 0.5,
            "x_max": 1.0,
            "y_min": 0.5,
            "y_max": 1.0,
            "grid_size": config.heatmap_grid_size
        },
        "timeline_steps": timeline_steps,
        "run": {
            "name": config.run_name or viewer_export_path.stem,
            "path": f"./runtime/{viewer_export_path.name}",
        },
    }
    viewer_export_path.write_text(json.dumps(payload, indent=2))
    return viewer_export_path


def update_viewer_manifest(
    manifest_path: Path,
    export_path: Path,
    config: Config,
    best_test_mse: float,
    timeline_steps: list[dict[str, object]],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"title": "HW1 QML Viewer Runs", "default_run": None, "runs": []}

    stem = export_path.stem
    label = config.run_name or stem
    entry = {
        "id": stem,
        "label": label,
        "path": f"./runtime/{export_path.name}",
        "steps": len(timeline_steps),
        "device": config.device_name,
        "diff_method": config.diff_method,
        "num_qubits": config.num_qubits,
        "num_layers": config.num_layers,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "best_test_mse": best_test_mse,
    }

    runs = [run for run in manifest.get("runs", []) if run.get("id") != stem]
    runs.insert(0, entry)
    manifest["runs"] = runs
    manifest["default_run"] = stem
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def train(config: Config, num_samples: int) -> None:
    torch.manual_seed(SEED)

    train_dataset, test_dataset = make_datasets(num_samples)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = DataReuploadingRegressor(
        num_qubits=config.num_qubits,
        num_layers=config.num_layers,
        hidden_scale=config.hidden_scale,
        device_name=config.device_name,
        diff_method=config.diff_method or resolve_diff_method(config.device_name, None),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    loss_history: list[dict[str, float]] = []
    timeline_steps: list[dict[str, object]] = []

    tracking_uri = config.tracking_uri or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    print("QCAA HW1 Problem 1 - data reuploading baseline")
    print(
        f"seed={SEED} samples={num_samples} qubits={config.num_qubits} "
        f"layers={config.num_layers} batch_size={config.batch_size} "
        f"epochs={config.epochs} lr={config.learning_rate}"
    , flush=True)
    print(f"device={config.device_name} diff_method={config.diff_method}", flush=True)
    print(f"trainable_parameters={trainable_parameters}", flush=True)
    print(f"mlflow_tracking_uri={tracking_uri}", flush=True)
    print(f"mlflow_experiment={config.experiment_name}", flush=True)
    print("viewer_steps=batch", flush=True)
    print(flush=True)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(
            {
                "seed": SEED,
                "num_samples": num_samples,
                "num_qubits": config.num_qubits,
                "num_layers": config.num_layers,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "hidden_scale": config.hidden_scale,
                "heatmap_grid_size": config.heatmap_grid_size,
                "device_name": config.device_name,
                "diff_method": config.diff_method,
                "viewer_export_path": config.viewer_export_path,
                "viewer_export_every": config.viewer_export_every,
                "trainable_parameters": trainable_parameters,
            }
        )

        global_step = 0
        for epoch in range(1, config.epochs + 1):
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
                progress.set_postfix(batch_loss=f"{batch_loss:.6f}", refresh=True)

                if global_step % config.viewer_export_every == 0:
                    test_mse = evaluate(model, test_loader, loss_fn)
                    heatmap_grids = build_heatmap_grids(model, config.heatmap_grid_size)
                    timeline_steps.append(
                        {
                            "label": f"epoch {epoch:02d} batch {batch_index:02d}",
                            "epoch": epoch,
                            "batch": batch_index,
                            "global_step": global_step,
                            "batch_loss": batch_loss,
                            "test_mse": test_mse,
                            "heatmaps": heatmap_grids,
                        }
                    )
                    mlflow.log_metric("batch_loss", batch_loss, step=global_step)
                    mlflow.log_metric("step_test_mse", test_mse, step=global_step)

            train_mse = evaluate(model, train_loader, loss_fn)
            test_mse = evaluate(model, test_loader, loss_fn)
            loss_history.append(
                {"epoch": float(epoch), "train_mse": train_mse, "test_mse": test_mse}
            )
            mlflow.log_metric("train_mse", train_mse, step=epoch)
            mlflow.log_metric("test_mse", test_mse, step=epoch)

            if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
                print(
                    f"epoch={epoch:02d} "
                    f"train_mse={train_mse:.6f} "
                    f"test_mse={test_mse:.6f}"
                , flush=True)

        best_test_mse = min(entry["test_mse"] for entry in loss_history)
        final_train_mse = loss_history[-1]["train_mse"]
        final_test_mse = loss_history[-1]["test_mse"]
        mlflow.log_metric("best_test_mse", best_test_mse)
        mlflow.log_metric("final_train_mse", final_train_mse)
        mlflow.log_metric("final_test_mse", final_test_mse)

        artifact_dir = Path("HW1") / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        hf_runtime_dir = Path("hf_space_hw1") / "runtime"
        hf_runtime_dir.mkdir(parents=True, exist_ok=True)
        viewer_export_path = resolve_viewer_export_path(config)
        viewer_manifest_path = Path("hf_space_hw1") / "runtime" / "viewer_manifest.json"
        loss_curve_path = artifact_dir / "problem1_loss_curve.png"
        circuit_path = artifact_dir / "problem1_circuit.png"
        heatmap_path = artifact_dir / "problem1_prediction_heatmap.png"
        hf_circuit_path = hf_runtime_dir / "problem1_circuit.png"
        final_heatmap_grids = timeline_steps[-1]["heatmaps"] if timeline_steps else build_heatmap_grids(
            model, config.heatmap_grid_size
        )
        make_loss_curve(loss_history, loss_curve_path)
        make_circuit_diagram(model, train_dataset[0][0], circuit_path)
        make_circuit_diagram(model, train_dataset[0][0], hf_circuit_path)
        make_prediction_heatmap(final_heatmap_grids, heatmap_path)
        write_viewer_export(config, viewer_export_path, timeline_steps)
        update_viewer_manifest(
            viewer_manifest_path,
            viewer_export_path,
            config,
            best_test_mse,
            timeline_steps,
        )
        mlflow.log_artifact(str(loss_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(circuit_path), artifact_path="plots")
        mlflow.log_artifact(str(heatmap_path), artifact_path="plots")
        mlflow.log_artifact(str(viewer_export_path), artifact_path="viewer")
        mlflow.log_artifact(str(viewer_manifest_path), artifact_path="viewer")

        print(flush=True)
        print(f"best_test_mse={best_test_mse:.6f}", flush=True)
        print(f"loss_curve={loss_curve_path}", flush=True)
        print(f"circuit_png={circuit_path}", flush=True)
        print(f"heatmap_png={heatmap_path}", flush=True)
        print(f"viewer_export={viewer_export_path}", flush=True)
        print(f"viewer_manifest={viewer_manifest_path}", flush=True)


def parse_args() -> tuple[Config, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--heatmap-grid-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="lightning.qubit",
        choices=("default.qubit", "lightning.qubit"),
    )
    parser.add_argument(
        "--diff-method",
        type=str,
        default=None,
        choices=("backprop", "adjoint"),
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--viewer-export-path", type=str, default=None)
    parser.add_argument("--viewer-export-every", type=int, default=1)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="hw1-problem1-datareuploading")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    diff_method = resolve_diff_method(args.device, args.diff_method)
    validate_device_config(args.device, diff_method)

    config = Config(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_scale=args.hidden_scale,
        heatmap_grid_size=args.heatmap_grid_size,
        device_name=args.device,
        diff_method=diff_method,
        viewer_export_path=args.viewer_export_path,
        viewer_export_every=args.viewer_export_every,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    return config, args.num_samples


def main() -> None:
    config, num_samples = parse_args()
    train(config, num_samples)


if __name__ == "__main__":
    main()
