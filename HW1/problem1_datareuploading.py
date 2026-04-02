"""First-pass PennyLane data reuploading baseline for QCAA HW1 Problem 1."""

from __future__ import annotations

import argparse
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
    epochs: int = 20
    learning_rate: float = 0.03
    hidden_scale: float = 1.0
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
    def __init__(self, num_qubits: int, num_layers: int, hidden_scale: float = 1.0) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.input_projection = nn.Linear(2, num_qubits)
        self.output_head = nn.Linear(num_qubits, 1)
        self.feature_scale = nn.Parameter(torch.full((num_qubits,), hidden_scale))
        self.weights = nn.Parameter(0.05 * torch.randn(num_layers, num_qubits, 3))

        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
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


def train(config: Config, num_samples: int) -> None:
    torch.manual_seed(SEED)

    train_dataset, test_dataset = make_datasets(num_samples)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = DataReuploadingRegressor(
        num_qubits=config.num_qubits,
        num_layers=config.num_layers,
        hidden_scale=config.hidden_scale,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    loss_history: list[dict[str, float]] = []

    tracking_uri = config.tracking_uri or f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    print("QCAA HW1 Problem 1 - data reuploading baseline")
    print(
        f"seed={SEED} samples={num_samples} qubits={config.num_qubits} "
        f"layers={config.num_layers} batch_size={config.batch_size} "
        f"epochs={config.epochs} lr={config.learning_rate}"
    , flush=True)
    print(f"trainable_parameters={trainable_parameters}", flush=True)
    print(f"mlflow_tracking_uri={tracking_uri}", flush=True)
    print(f"mlflow_experiment={config.experiment_name}", flush=True)
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
                "trainable_parameters": trainable_parameters,
            }
        )

        for epoch in range(1, config.epochs + 1):
            model.train()
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"epoch {epoch:02d}/{config.epochs:02d}",
                leave=False,
                dynamic_ncols=True,
            )

            for features, labels in progress:
                optimizer.zero_grad()
                predictions = model(features)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                progress.set_postfix(batch_loss=f"{float(loss.item()):.6f}", refresh=True)

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
        loss_curve_path = artifact_dir / "problem1_loss_curve.png"
        circuit_path = artifact_dir / "problem1_circuit.png"
        make_loss_curve(loss_history, loss_curve_path)
        make_circuit_diagram(model, train_dataset[0][0], circuit_path)
        mlflow.log_artifact(str(loss_curve_path), artifact_path="plots")
        mlflow.log_artifact(str(circuit_path), artifact_path="plots")

        print(flush=True)
        print(f"best_test_mse={best_test_mse:.6f}", flush=True)
        print(f"loss_curve={loss_curve_path}", flush=True)
        print(f"circuit_png={circuit_path}", flush=True)


def parse_args() -> tuple[Config, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="hw1-problem1-datareuploading")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    config = Config(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_scale=args.hidden_scale,
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
