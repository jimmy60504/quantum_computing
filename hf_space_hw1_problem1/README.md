---
title: Data Reuploading Results Bundle
emoji: 📈
colorFrom: yellow
colorTo: green
sdk: static
pinned: false
---

# Data Reuploading Results Bundle

This directory is a static Hugging Face package for `HW1 Problem 1`, but it is
not only a viewer. The published bundle is meant to carry three things together:

1. The experiment results exported from `gx10`
2. A lightweight static UI for inspecting those results
3. A source snapshot so readers can see how the quantum model was built

The emphasis is the quantum experiment itself: what circuit family was tested,
what options were compared, and how those choices affect the model.

## What this experiment is about

The task is regression on:

```text
f(x1, x2) = sin(exp(x1) + x2)
```

The split is intentionally hard:

- train domain: `[0.0, 0.5] x [0.0, 0.5]`
- test domain: `[0.5, 1.0] x [0.5, 1.0]`

So this is not a simple interpolation exercise. The model sees one region and
is then asked to generalize into a different region with a noticeably different
surface shape. That is why the interesting question is not just whether train
loss goes down, but whether the quantum model carries useful structure into the
held-out test domain.

## What is inside the published bundle

When prepared through `gx10`, the output bundle contains both the app and a
source export:

- `runtime/`: generated experiment outputs such as viewer exports, metrics, and
  circuit artifacts
- `source/README.md`: top-level project README from the experiment repo
- `source/environment.yml`: local Conda environment definition
- `source/Dockerfile`: remote `gx10` container definition
- `source/HW1/problem1/`: the core Problem 1 source code
- `source/scripts/`: the main `gx10` helper scripts used to run and package the experiment

So the Hugging Face package acts more like a compact experiment bundle than a
UI-only artifact.

## Quantum model: the important code pieces

The key source files in the bundled `source/HW1/problem1/` tree are:

- `datareuploading.py`: CLI entry point and experiment configuration
- `core/modeling.py`: dataset generation, classical encoding, quantum circuit,
  and regressor definition
- `core/training.py`: optimization loop, snapshot export, MLflow logging
- `tools/evaluate_snapshot_chunk.py`: post-process per-step train/test MSE
- `tools/render_snapshot_chunk.py`: post-process per-step surfaces and heatmaps
- `tools/fourier_analysis.py`: frequency-domain analysis of the final export

The actual model is a hybrid regressor:

1. Classical input features are optionally lifted before entering the quantum model.
2. A learned linear projection maps those lifted features onto `num_qubits`.
3. Each layer applies data reuploading with `RY` and `RZ` on every wire.
4. A ring of `CNOT` gates entangles the wires.
5. Trainable `Rot` gates provide the variational parameters.
6. The circuit returns one `PauliZ` expectation value per qubit.
7. A classical output head maps those quantum features to the scalar prediction.

That means the three main experimental levers are:

- feature encoding
- number of qubits
- number of reuploading layers

## What options were actually tested

The current standard pipeline tests this matrix:

- `raw`, `q=2`, `l=2`
- `raw`, `q=2`, `l=3`
- `raw`, `q=3`, `l=2`
- `raw`, `q=3`, `l=3`
- `poly`, `q=2`, `l=2`
- `exp`, `q=2`, `l=2`

Those options differ in meaningful ways:

- `raw`: passes `[x1, x2]` directly into the learned projection
- `poly`: expands to `[x1, x2, x1^2, x1*x2, x2^2]` before projection
- `exp`: uses `[exp(x1), x2]`, which injects a feature closer to the target function structure
- `q=2` vs `q=3`: changes the number of wires, the number of circuit outputs, and the size of the trainable quantum block
- `l=2` vs `l=3`: changes how many times data reuploading, entanglement, and trainable rotations are repeated

So the comparison is not just "small vs large." It is also:

- how much classical inductive bias we provide before the circuit
- how much quantum capacity we allocate in wires and repeated layers
- whether extra capacity helps extrapolation, or only improves fit on the train region

## What the static app shows

The viewer is still useful, but it is only one part of the bundle. It helps
inspect:

- final train and test MSE across runs
- per-step train/test surfaces
- absolute error heatmaps
- the circuit diagram for the selected run
- a Fourier summary for the final prediction surface

If a runtime manifest is present, the app reads `runtime/viewer_manifest.json`.
If not, it falls back to the template data under `data/`.

## Local preview

Serve the folder as a static site:

```bash
cd hf_space_hw1_problem1
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## How the bundle is prepared on `gx10`

From the experiment repo on `gx10`:

```bash
cd ~/quantum_computing
./scripts/gx10_prepare_hf_space.sh
```

This creates a clean publish directory under:

```text
.out/hf_space_hw1_problem1_publish/
```

The script copies:

- the static app scaffold from `hf_space_hw1_problem1/`
- the latest runtime exports from `hf_space_hw1_problem1/runtime/`
- a curated source snapshot under `source/`

That publish directory is the one to push to a Hugging Face Static Space repo.
