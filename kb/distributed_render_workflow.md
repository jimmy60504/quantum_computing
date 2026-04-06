# Distributed Render Workflow

This note describes a reusable pattern for training on one machine and
rendering per-step viewer artifacts later from saved snapshots.

## Why this exists

Some QML experiments are fast enough to train but slow to visualize because
every saved step must rebuild a full prediction surface or heatmap. When the
viewer needs every step, the render phase can dominate total runtime.

For `HW1/problem1`, the tuned `gx10` single-machine path is now the default
because it outperformed the earlier distributed helper setup. This note remains
useful as a design reference and fallback pattern.

The workflow here splits the job into four phases:

1. Train once and export per-step model snapshots.
2. Evaluate disjoint step ranges into lightweight metrics chunks.
3. Render disjoint step ranges into full viewer chunks only when needed.
4. Merge the chosen chunks back into a summary export or final viewer export.

## Reusable contract

For a homework problem to use this workflow cleanly, it should provide:

- A snapshot export JSON with:
  - static viewer metadata
  - sample points or other immutable plot inputs
  - render configuration
  - a `timeline_snapshots` list that stores per-step model state
- A render function with the shape:
  - `(config, snapshot) -> rendered_timeline_step`
- A merge step that sorts rendered steps by `global_step` and updates the
  viewer manifest from the merged result

This keeps the transport and orchestration generic while letting each problem
define its own render logic.

## Problem 1 prototype

Problem 1 now supports two training modes:

- `--render-mode inline`
  - train and render immediately, like the original workflow
- `--render-mode snapshots-only`
  - train and export step snapshots without rendering them inline

Relevant scripts:

- `HW1/problem1/datareuploading.py`
- `HW1/problem1/tools/evaluate_snapshot_chunk.py`
- `HW1/problem1/tools/merge_evaluated_chunks.py`
- `HW1/problem1/tools/render_snapshot_chunk.py`
- `HW1/problem1/tools/merge_rendered_chunks.py`

### Training on `gx10`

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --render-mode snapshots-only \
  --encoding phase_learnable \
  --num-qubits 1 \
  --num-layers 1 \
  --epochs 20 \
  --viewer-export-every 1 \
  --run-name phase-learnable-q1-l1-e20
```

This produces a snapshot export such as:

- `HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json`

In `snapshots-only` mode the training loop no longer computes per-epoch
train/test MSE inline. Those metrics are reconstructed later from the saved
model states.

### Optional helper-machine mount

If you ever do need helper machines again, mount the `gx10` repo manually with
your preferred `sshfs` command or copy only the snapshot export plus repo code.
The repository no longer ships dedicated mount helper scripts because the
current default workflow does not need them.

### Evaluating metrics for a chunk

```bash
mkdir -p ~/metrics_chunks
cd ~/mnt/gx10-quantum
python3 HW1/problem1/tools/evaluate_snapshot_chunk.py \
  --snapshot-export ~/mnt/gx10-quantum/HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json \
  --start-index 0 \
  --end-index 80 \
  --output ~/metrics_chunks/phase-learnable-q1-l1-e20_metrics_chunk_00000_00079.json
```

Merge the metrics chunks back into a lightweight summary export:

```bash
cd ~/quantum_computing
python3 HW1/problem1/tools/merge_evaluated_chunks.py \
  --snapshot-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json \
  --chunk-glob "HW1/problem1/hf_space/runtime/metrics/phase-learnable-q1-l1-e20_snapshots_metrics_chunk_*.json"
```

This gives a cheap way to compare `train_mse` and `test_mse` across many runs
before paying for full heatmap rendering.

### Rendering a chunk

```bash
mkdir -p ~/render_chunks
cd ~/mnt/gx10-quantum
python3 HW1/problem1/tools/render_snapshot_chunk.py \
  --snapshot-export ~/mnt/gx10-quantum/HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json \
  --start-index 0 \
  --end-index 80 \
  --output ~/render_chunks/phase-learnable-q1-l1-e20_chunk_00000_00079.json
```

If the helper machine needs a more portable simulator, override the render
backend:

```bash
python3 HW1/problem1/tools/render_snapshot_chunk.py \
  --snapshot-export ... \
  --start-index 80 \
  --end-index 160 \
  --device default.qubit \
  --diff-method backprop
```

### Merging chunks on `gx10`

Once chunk JSON files are copied back, merge them into the final viewer export:

```bash
cd ~/quantum_computing
python3 HW1/problem1/tools/merge_rendered_chunks.py \
  --snapshot-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20_snapshots.json \
  --chunk-glob "HW1/problem1/hf_space/runtime/chunks/phase-learnable-q1-l1-e20_snapshots_chunk_*.json"
```

Then run Fourier analysis on the merged viewer export:

```bash
cd ~/quantum_computing
python3 HW1/problem1/tools/fourier_analysis.py \
  --viewer-export HW1/problem1/hf_space/runtime/phase-learnable-q1-l1-e20.json
```

## Safety rules

- Treat the shared snapshot export directory as read-mostly.
- Let helper machines write chunk files, not the final viewer JSON.
- Do the final merge on a single machine to avoid manifest or export conflicts.

## How to extend this to other HW problems

If another problem needs per-step visual playback:

1. Export step-local model state plus static plotting inputs.
2. Implement a deterministic chunk renderer for one snapshot at a time.
3. Add a cheap metrics-only pass when selection can be done without heatmaps.
4. If helper machines are needed, keep merge as a single-machine step.

The more the problem can separate immutable inputs from step-local state, the
easier it is to fan render work out across multiple CPUs and machines.
