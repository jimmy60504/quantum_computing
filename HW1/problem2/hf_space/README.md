---
title: QML Classifier Explorer
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# QML Classifier Explorer

Static Hugging Face viewer for **HW1 Problem 2** — comparing three quantum
machine learning classification methods on two datasets.

## What this experiment is about

Three QML approaches are evaluated on binary classification:

| Method | Description |
|--------|-------------|
| **Explicit Quantum Model** | Encoding circuit S(x) + trainable W(θ) + measurement |
| **Implicit Quantum Kernel** | Fixed encoding kernel k(xi, xj) passed to SVM |
| **Data Reuploading** | Interleaved encoding and trainable layers (Ref. [4]) |

Two datasets:
- **Circle** — concentric ring structure (as used in Ref. [4])
- **Moons** — `sklearn.datasets.make_moons(noise=0.1, n_samples=200)`

## Viewer contents

1. **Decision boundary grid** — 3 methods × 2 datasets (6 plots)
2. **Training curve** — accuracy and loss vs epoch, with step slider
3. **Comparison table** — test accuracy, trainable parameters, training time

## Generating runtime data

Run on `gx10`:

```bash
# training + export
ssh gx10 "cd ~/quantum_computing && GX10_DOCKER_NETWORK=gx10-mlflow ./scripts/gx10_run_py.sh HW1/problem2/train.py --run-name q2-l4-e50 --tracking-uri http://gx10-mlflow-server:5001"
```

The export populates `runtime/viewer_data.json` which the viewer picks up
automatically. Without a runtime export the viewer shows the template
placeholder.
