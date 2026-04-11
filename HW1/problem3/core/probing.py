"""Per-sample prediction probing for HW1 Problem 3.

Provides three capabilities:
1. export_gallery — save candidate test images as PNGs for manual selection
2. probe_checkpoints — run selected samples through saved checkpoints
3. probe_tsne — extract backbone features at checkpoints and run t-SNE/UMAP
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from .config import Prob3Config
from .datasets import CLASS_NAMES, CIFAR10_MEAN, CIFAR10_STD, get_transforms
from .training import _build_model


def _raw_test_dataset(data_dir: str | Path = "./data/cifar10") -> datasets.CIFAR10:
    """Load CIFAR-10 test set without transforms (returns PIL images)."""
    return datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=None,
    )


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def export_gallery(
    output_dir: Path,
    data_dir: str | Path = "./data/cifar10",
    per_class: int = 10,
) -> Path:
    """Save first `per_class` test images per class as PNGs to output_dir/gallery/."""
    raw_ds = _raw_test_dataset(data_dir)
    gallery_dir = output_dir / "gallery"

    counts: dict[int, int] = {}
    for idx in range(len(raw_ds)):
        img, label = raw_ds[idx]
        if counts.get(label, 0) >= per_class:
            continue
        counts[label] = counts.get(label, 0) + 1
        class_name = CLASS_NAMES[label]
        class_dir = gallery_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        img.save(class_dir / f"idx_{idx:05d}.png")

    total = sum(counts.values())
    print(f"Gallery exported: {total} images → {gallery_dir}")
    return gallery_dir


def _select_samples(
    data_dir: str | Path,
    sample_indices: list[int] | None,
) -> list[dict]:
    """Load selected test samples. If no indices given, pick first per class."""
    raw_ds = _raw_test_dataset(data_dir)
    test_transform = get_transforms(train=False)

    if sample_indices is None:
        # Default: first occurrence of each class
        seen: set[int] = set()
        sample_indices = []
        for idx in range(len(raw_ds)):
            _, label = raw_ds[idx]
            if label not in seen:
                seen.add(label)
                sample_indices.append(idx)
            if len(seen) == len(CLASS_NAMES):
                break

    samples = []
    for idx in sample_indices:
        pil_img, label = raw_ds[idx]
        tensor_img = test_transform(pil_img)
        samples.append({
            "dataset_index": idx,
            "class_idx": label,
            "class_name": CLASS_NAMES[label],
            "image_base64": _pil_to_base64(pil_img),
            "image_tensor": tensor_img,
        })

    return samples


def _parse_step(filename: str) -> int | None:
    m = re.search(r"step(\d+)", filename)
    return int(m.group(1)) if m else None


def _probe_worker(
    method: str,
    config: Prob3Config,
    device_str: str,
    ckpts: list[tuple[int, Path]],
    sample_batch: torch.Tensor,
) -> list[dict]:
    """Worker: build model once, loop through a slice of checkpoints."""
    device = torch.device(device_str)
    sample_batch = sample_batch.to(device)
    model = _build_model(method, config).to(device)
    model.eval()

    results = []
    for step, ckpt_path in ckpts:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        with torch.no_grad():
            logits = model(sample_batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        results.append({
            "step": step,
            "probs": [[round(float(p), 6) for p in row] for row in probs],
        })
    return results


def probe_checkpoints(
    config: Prob3Config,
    run_dir: Path,
    sample_indices: list[int] | None = None,
    data_dir: str | Path = "./data/cifar10",
    methods: list[str] | None = None,
) -> Path:
    """Run selected samples through all checkpoints and save probe artifact."""
    device_str = config.device
    samples = _select_samples(data_dir, sample_indices)

    # Stack sample tensors (kept on CPU for sharing across workers)
    sample_batch = torch.stack([s["image_tensor"] for s in samples])

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory: {ckpt_dir}")

    # Discover methods and their checkpoints
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    method_ckpts: dict[str, list[tuple[int, Path]]] = {}
    for f in ckpt_files:
        parts = f.stem.split("_step")
        if len(parts) != 2:
            continue
        method = parts[0]
        step = _parse_step(f.stem)
        if step is not None:
            method_ckpts.setdefault(method, []).append((step, f))

    for method in method_ckpts:
        method_ckpts[method].sort(key=lambda x: x[0])

    # Filter to requested methods
    if methods:
        method_ckpts = {m: v for m, v in method_ckpts.items() if m in methods}

    # Probe each method:
    # - MLP: parallel with ProcessPoolExecutor (fork, CPU-only, safe)
    # - QNN: sequential (PennyLane lightning.gpu is not fork-safe across workers)
    probes: dict[str, list[dict]] = {}
    num_workers = max(1, min(os.cpu_count() or 4, 8))

    for method, ckpts in method_ckpts.items():
        t0 = time.time()
        if method == "mlp":
            # Parallel: split into slices, one per worker
            print(f"[probe] {method}: {len(ckpts)} checkpoints, {num_workers} workers (parallel)")
            slices = [ckpts[i::num_workers] for i in range(num_workers)]
            slices = [s for s in slices if s]
            worker_args = [
                (method, config, device_str, sl, sample_batch.clone())
                for sl in slices
            ]
            with ProcessPoolExecutor(max_workers=len(slices)) as pool:
                futures = {pool.submit(_probe_worker, *args): i for i, args in enumerate(worker_args)}
                slice_results = [None] * len(slices)
                for fut in as_completed(futures):
                    slice_results[futures[fut]] = fut.result()
            method_probes = [entry for result in slice_results for entry in result]
        else:
            # Sequential: single model instance, iterate checkpoints
            print(f"[probe] {method}: {len(ckpts)} checkpoints, sequential")
            method_probes = _probe_worker(method, config, device_str, ckpts, sample_batch.clone())

        method_probes.sort(key=lambda x: x["step"])
        elapsed = time.time() - t0
        print(f"[probe] {method}: done in {elapsed:.1f}s")
        probes[method] = method_probes

    # Build artifact
    artifact = {
        "samples": [
            {
                "dataset_index": s["dataset_index"],
                "class_idx": s["class_idx"],
                "class_name": s["class_name"],
                "image_base64": s["image_base64"],
            }
            for s in samples
        ],
        "probes": probes,
    }

    artifact_path = run_dir / "probe_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Probe artifact → {artifact_path}")
    return artifact_path


# ── t-SNE / UMAP feature-space probing ───────────────────────────────────────

def _select_tsne_samples(
    data_dir: str | Path,
    n_per_class: int,
) -> list[dict]:
    """Load n_per_class samples per class from test set, with base64 thumbnails."""
    raw_ds = _raw_test_dataset(data_dir)
    test_transform = get_transforms(train=False)

    counts: dict[int, int] = {}
    samples: list[dict] = []
    for idx in range(len(raw_ds)):
        pil_img, label = raw_ds[idx]
        if counts.get(label, 0) >= n_per_class:
            continue
        counts[label] = counts.get(label, 0) + 1
        samples.append({
            "dataset_index": idx,
            "class_idx": label,
            "class_name": CLASS_NAMES[label],
            "image_base64": _pil_to_base64(pil_img),
            "image_tensor": test_transform(pil_img),
        })
        if len(samples) == n_per_class * len(CLASS_NAMES):
            break
    return samples


def _select_train_samples(
    data_dir: str | Path,
    n_per_class: int,
) -> tuple[torch.Tensor, list[int]]:
    """Load n_per_class samples per class from training set (no images needed)."""
    raw_ds = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=None,
    )
    train_transform = get_transforms(train=False)  # eval normalisation, no augment

    counts: dict[int, int] = {}
    tensors: list[torch.Tensor] = []
    labels: list[int] = []
    for idx in range(len(raw_ds)):
        pil_img, label = raw_ds[idx]
        if counts.get(label, 0) >= n_per_class:
            continue
        counts[label] = counts.get(label, 0) + 1
        tensors.append(train_transform(pil_img))
        labels.append(label)
        if len(tensors) == n_per_class * len(CLASS_NAMES):
            break
    return torch.stack(tensors), labels


def _extract_backbone_features(
    model: torch.nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run model.backbone forward pass and return numpy array [N, feature_dim]."""
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        feats = model.backbone(sample_batch)
    return feats.cpu().numpy()


def _run_reduction(
    features_joint: np.ndarray,
    reduction: str = "tsne",
) -> tuple[np.ndarray, str]:
    """Dimensionality reduction → 2D. Returns (coords [n_total, 2], name)."""
    if reduction == "umap":
        try:
            import umap as umap_lib
            print(f"[tsne] UMAP on {features_joint.shape}...")
            reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(features_joint)
            return coords, "umap"
        except ImportError:
            print("[tsne] umap-learn not available, falling back to t-SNE")
            reduction = "tsne"

    if reduction == "pca":
        from sklearn.decomposition import PCA
        print(f"[tsne] PCA on {features_joint.shape}...")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(features_joint)
        return coords, "pca"

    # Default: t-SNE with PCA pre-reduction for speed
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_total, n_feat = features_joint.shape
    pca_dim = min(50, n_feat, n_total - 1)
    if n_feat > pca_dim:
        print(f"[tsne] PCA pre-reduction {n_feat}→{pca_dim}...")
        features_joint = PCA(n_components=pca_dim, random_state=42).fit_transform(features_joint)

    perplexity = min(30.0, max(5.0, n_total / 10))
    print(f"[tsne] t-SNE on {features_joint.shape} (perplexity={perplexity:.0f})...")
    coords = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        n_iter=1000,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features_joint)
    return coords, "tsne"


def _make_anchors(n_classes: int) -> np.ndarray:
    """Unit-circle anchor positions for each class, evenly spaced. [C, 2]"""
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _simplex_project(probs_list: list[np.ndarray]) -> list[list[list[float]]]:
    """Project probability vectors via class anchors on the unit circle.

    position = softmax_probs @ anchors  →  weighted centroid in 2D.
    Uniform predictions land at origin; confident predictions reach class anchor.
    """
    anchors = _make_anchors(probs_list[0].shape[1])
    return [
        [[round(float(x), 4), round(float(y), 4)] for x, y in (p @ anchors)]
        for p in probs_list
    ]


def _compute_class_centroids(
    model: torch.nn.Module,
    data_dir: str | Path,
    device: torch.device,
    n_classes: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Run full test set through model; return per-class mean 2D position. [C, 2]"""
    from .datasets import get_transforms
    import torchvision

    test_ds = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True,
        transform=get_transforms(train=False),
    )
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    anchors = torch.tensor(_make_anchors(n_classes), dtype=torch.float32, device=device)

    sum_2d = torch.zeros(n_classes, 2, device=device)
    counts = torch.zeros(n_classes, device=device)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs = torch.softmax(model(images), dim=1)   # [B, C]
            coords = probs @ anchors                       # [B, 2]
            sum_2d.index_add_(0, labels, coords)
            counts.index_add_(0, labels, torch.ones(len(labels), device=device))

    centroids = (sum_2d / counts.unsqueeze(1)).cpu().numpy()  # [C, 2]
    return centroids


def probe_tsne(
    config: Prob3Config,
    run_dir: Path,
    n_per_class: int = 100,
    n_steps: int = 400,
    data_dir: str | Path = "./data/cifar10",
    methods: list[str] | None = None,
    reduction: str = "umap",
) -> Path:
    """Build the feature-space evolution artifact for the viewer.

    Strategy (handles both frozen and unfrozen backbone):
      1. Extract backbone features from the FINAL checkpoint → UMAP/t-SNE once
         → fixed 2D coordinates that define the "landscape".
      2. At each selected step, run the FULL forward pass → softmax predictions.
         Store predicted_class + confidence for every sample.

    Viewer animation: dots at fixed positions, colours change from random
    predictions (step 1) to correct-class colours (final step). Frozen backbone?
    The landscape is static but the colour animation is still compelling. Trainable
    backbone? Both positions AND colours evolve.
    """
    device = torch.device(config.device)
    samples = _select_tsne_samples(data_dir, n_per_class)
    n_samples = len(samples)
    print(f"[tsne] {n_samples} samples ({n_per_class}/class × {len(CLASS_NAMES)} classes)")

    sample_batch = torch.stack([s["image_tensor"] for s in samples])

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory: {ckpt_dir}")

    # Discover checkpoints by method
    method_ckpts: dict[str, list[tuple[int, Path]]] = {}
    for f in sorted(ckpt_dir.glob("*.pt")):
        parts = f.stem.split("_step")
        if len(parts) != 2:
            continue
        m = parts[0]
        step = _parse_step(f.stem)
        if step is not None:
            method_ckpts.setdefault(m, []).append((step, f))
    for m in method_ckpts:
        method_ckpts[m].sort(key=lambda x: x[0])

    if methods:
        method_ckpts = {m: v for m, v in method_ckpts.items() if m in methods}

    all_method_data: dict[str, dict] = {}

    for method, all_ckpts in method_ckpts.items():
        total = len(all_ckpts)
        # Checkpoints are now uniform across all epochs (checkpoint_freq per epoch),
        # so plain linspace gives equal airtime to every epoch.
        idxs = sorted(set(
            np.linspace(0, total - 1, min(n_steps, total), dtype=int).tolist()
        ))
        selected_ckpts = [all_ckpts[i] for i in idxs]
        selected_steps = [c[0] for c in selected_ckpts]
        n_sel = len(selected_ckpts)
        print(f"[tsne] {method}: {n_sel} frames out of {total} checkpoints")

        model = _build_model(method, config).to(device)
        model.eval()

        # ── Collect scatter probs at selected checkpoints ─────────────────────
        probs_list: list[np.ndarray] = []   # each: [N, 10]
        preds_per_step: list[list[int]] = []
        t0 = time.time()
        for i, (step, ckpt_path) in enumerate(selected_ckpts):
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            with torch.no_grad():
                logits = model(sample_batch.to(device))
                probs = F.softmax(logits, dim=1).cpu()
            probs_np = probs.numpy()                              # [N, 10]
            probs_list.append(probs_np)
            preds_per_step.append(probs.argmax(dim=1).tolist())  # [N] ints
            if (i + 1) % 40 == 0 or i == n_sel - 1:
                print(f"[tsne] {method}: {i+1}/{n_sel} checkpoints loaded")

        print(f"[tsne] {method}: forward passes in {time.time()-t0:.1f}s")

        # ── Project onto class-anchor simplex (no fitting needed) ────────────
        t1 = time.time()
        coords_per_step = _simplex_project(probs_list)
        print(f"[tsne] {method}: simplex projection in {time.time()-t1:.1f}s")

        # ── Class centroids from full test set at final checkpoint ────────────
        # Load final checkpoint again, run all 10k test images, average per class.
        print(f"[tsne] {method}: computing class centroids on full test set...")
        last_ckpt = selected_ckpts[-1][1]
        model.load_state_dict(torch.load(last_ckpt, map_location=device, weights_only=True))
        centroids_2d = _compute_class_centroids(model, data_dir, device, len(CLASS_NAMES))
        class_centroids = [
            {
                "class_idx": i,
                "class_name": CLASS_NAMES[i],
                "x": round(float(centroids_2d[i, 0]), 4),
                "y": round(float(centroids_2d[i, 1]), 4),
            }
            for i in range(len(CLASS_NAMES))
        ]
        print(f"[tsne] {method}: centroids done")

        all_method_data[method] = {
            "reduction": "simplex",
            "steps": selected_steps,
            "coords": coords_per_step,
            "preds": preds_per_step,
            "class_centroids": class_centroids,
        }

    artifact = {
        "samples": [
            {"dataset_index": s["dataset_index"], "class_idx": s["class_idx"],
             "class_name": s["class_name"], "image_base64": s["image_base64"]}
            for s in samples
        ],
        "n_per_class": n_per_class,
        "methods": all_method_data,
    }

    artifact_path = run_dir / "tsne_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"t-SNE artifact → {artifact_path}")
    return artifact_path
