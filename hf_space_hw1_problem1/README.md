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

The current model family is intentionally more structured than the earlier
generic baseline. It starts from the exact circuit identity for this task and
then relaxes assumptions one step at a time:

1. `quantum_exact`
2. `phase_learnable`
3. `scaled_exact`

The shared core idea is:

1. Upload `exp(x1)` and `x2` on the same rotation axis.
2. Let same-axis reuploading add those angles.
3. Use a phase shift to turn the measured `PauliZ` expectation from cosine into sine.

This keeps the architecture tied to the problem structure instead of asking a
generic projection-heavy ansatz to rediscover it from scratch.

## What options were actually tested

The current standard pipeline tests the structured generalization ladder:

- `quantum_exact`, `q=1`, `l=1`
- `phase_learnable`, `q=1`, `l=1`
- `scaled_exact`, `q=1`, `l=1`

Those options differ in a deliberately staged way:

- `quantum_exact`: fixed constructive solution for `sin(exp(x1) + x2)`
- `phase_learnable`: keeps the same circuit skeleton but learns the phase shift
- `scaled_exact`: additionally learns scale and bias terms on `exp(x1)` and `x2`

So the comparison is no longer "generic small vs generic large." It is:

- whether the exact problem-aligned circuit already solves the task
- whether the phase can be learned instead of fixed
- whether a slightly relaxed structured circuit still preserves good generalization

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

To prepare and upload in one step:

```bash
cd ~/quantum_computing
HF_SPACE_RUNTIME_DATASET_REPO=jimmy60504/data-reuploading-demo-runtime-test \
./scripts/gx10_upload_hf_space.sh jimmy60504/Data-Reuploading-Demo
```

The Space upload uses `hf upload`, which is more reliable here than
`upload-large-folder` for the small static bundle. The runtime payload can stay
in a public dataset repo and be fetched by the browser at display time.
