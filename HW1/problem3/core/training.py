"""Training loop for HW1 Problem 3 — CNN+MLP baseline and CNN+QNN hybrid."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Prob3Config, resolve_viewer_export_path, resolve_viewer_manifest_path
from .datasets import get_dataloaders, CLASS_NAMES
from .models import CNNMLPClassifier, CNNQNNClassifier
from .viewer_io import build_viewer_payload, write_viewer_export, update_viewer_manifest


def _build_model(method: str, config: Prob3Config) -> nn.Module:
    torch.manual_seed(config.seed)
    if method == "mlp":
        return CNNMLPClassifier(
            feature_dim=config.feature_dim,
            freeze_backbone=config.freeze_backbone,
        )
    elif method == "qnn":
        return CNNQNNClassifier(
            feature_dim=config.feature_dim,
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            freeze_backbone=config.freeze_backbone,
            device_name=config.q_device_name,
            diff_method=config.q_diff_method,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> dict[str, float | list]:
    """Evaluate model and return loss, accuracy, and confusion matrix."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_seen += labels.size(0)

        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "confusion": confusion.tolist(),
    }


def train_method(method: str, config: Prob3Config, run_dir: Path | None = None) -> dict:
    """Train one method (mlp or qnn) and return training history + final metrics."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_loader, test_loader = get_dataloaders(
        batch_size=config.batch_size,
        seed=config.seed,
    )

    model = _build_model(method, config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    num_params = model.count_trainable_params()
    print(f"[{method}] Trainable parameters: {num_params:,}")

    history: list[dict] = []
    best_test_acc = 0.0
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{method}] Epoch {epoch}/{config.epochs}",
            leave=False,
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_seen += batch_size

            pbar.set_postfix(
                loss=f"{running_loss / running_seen:.4f}",
                acc=f"{running_correct / running_seen:.3f}",
            )

        scheduler.step()

        train_loss = running_loss / max(running_seen, 1)
        train_acc = running_correct / max(running_seen, 1)

        # ── Evaluate ──
        test_metrics = evaluate(model, test_loader, device)
        test_loss = test_metrics["loss"]
        test_acc = test_metrics["accuracy"]
        confusion = test_metrics["confusion"]
        best_test_acc = max(best_test_acc, test_acc)

        print(
            f"[{method}] Epoch {epoch}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.3f}  "
            f"best={best_test_acc:.3f}"
        )

        step_record = {
            "epoch": epoch,
            "global_step": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
            "label": f"Epoch {epoch}",
        }

        # Include confusion matrix at export epochs
        if epoch % config.viewer_export_every == 0 or epoch == config.epochs:
            step_record["confusion"] = confusion

        history.append(step_record)

    elapsed = time.time() - t0
    print(f"[{method}] Training complete in {elapsed:.1f}s  best_test_acc={best_test_acc:.3f}")

    # ── Save artifact ──
    artifact = {
        "method": method,
        "method_label": model.method_label,
        "num_params": num_params,
        "train_time_s": round(elapsed, 1),
        "best_test_acc": best_test_acc,
        "history": history,
    }

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = run_dir / f"{method}_artifact.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"Saved artifact → {artifact_path}")

    return artifact


def assemble_viewer(config: Prob3Config, run_dir: Path) -> Path:
    """Combine MLP and QNN artifacts into viewer JSON."""
    artifacts: dict[str, dict] = {}
    for method in ("mlp", "qnn"):
        path = run_dir / f"{method}_artifact.json"
        if path.exists():
            artifacts[method] = json.loads(path.read_text())
        else:
            print(f"Warning: {path} not found, skipping {method}")

    if not artifacts:
        raise RuntimeError("No artifacts found to assemble.")

    export_path = resolve_viewer_export_path(config)
    manifest_path = resolve_viewer_manifest_path(config)

    write_viewer_export(config, export_path, artifacts)
    print(f"Viewer export → {export_path}")

    best_acc = max(
        (a.get("best_test_acc", 0) for a in artifacts.values()),
        default=0.0,
    )
    update_viewer_manifest(manifest_path, export_path, config, best_acc, artifacts)
    print(f"Manifest → {manifest_path}")

    return export_path


def train_all(config: Prob3Config) -> None:
    """Full pipeline: train both methods, then assemble viewer."""
    run_dir = Path(config.viewer_export_path) / "runs" / (config.run_name or "default")
    run_dir.mkdir(parents=True, exist_ok=True)

    for method in ("mlp", "qnn"):
        train_method(method, config, run_dir)

    assemble_viewer(config, run_dir)
