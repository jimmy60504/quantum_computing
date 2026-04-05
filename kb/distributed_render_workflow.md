# Distributed Render Workflow

This note describes a reusable pattern for training on one machine and
rendering per-step viewer artifacts on one or more helper machines.

## Why this exists

Some QML experiments are fast enough to train but slow to visualize because
every saved step must rebuild a full prediction surface or heatmap. When the
viewer needs every step, the render phase can dominate total runtime.

The workflow here splits the job into three phases:

1. Train once and export per-step model snapshots.
2. Render disjoint step ranges on helper machines.
3. Merge rendered chunks back into the final viewer export.

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
- `HW1/problem1/tools/render_snapshot_chunk.py`
- `HW1/problem1/tools/merge_rendered_chunks.py`

### Training on `gx10`

```bash
cd ~/quantum_computing
./scripts/gx10_run_py.sh HW1/problem1/datareuploading.py \
  --render-mode snapshots-only \
  --encoding raw \
  --num-qubits 2 \
  --num-layers 2 \
  --epochs 20 \
  --viewer-export-every 1 \
  --run-name raw-q2-l2-e20
```

This produces a snapshot export such as:

- `hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json`

### Mounting `gx10` data over `sshfs`

On a helper machine such as the Steam Deck:

```bash
mkdir -p ~/mnt
./scripts/mount_remote_dir_sshfs.sh gx10 /home/jimmy/quantum_computing ~/mnt/gx10-quantum
```

Unmount when done:

```bash
./scripts/unmount_remote_dir.sh ~/mnt/gx10-quantum
```

This mount can be the entire repo, so the helper machine does not need a second
copy of the codebase. In that setup, run the render script from the mounted repo
and write chunk outputs to a local writable directory.

### Rendering a chunk

```bash
mkdir -p ~/render_chunks
cd ~/mnt/gx10-quantum
python3 HW1/problem1/tools/render_snapshot_chunk.py \
  --snapshot-export ~/mnt/gx10-quantum/hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json \
  --start-index 0 \
  --end-index 80 \
  --output ~/render_chunks/raw-q2-l2-e20_chunk_00000_00079.json
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
  --snapshot-export hf_space_hw1_problem1/runtime/raw-q2-l2-e20_snapshots.json \
  --chunk-glob "hf_space_hw1_problem1/runtime/chunks/raw-q2-l2-e20_snapshots_chunk_*.json"
```

## Safety rules

- Treat the mounted `gx10` export directory as read-mostly.
- Let helper machines write chunk files, not the final viewer JSON.
- Do the final merge on a single machine to avoid manifest or export conflicts.

## How to extend this to other HW problems

If another problem needs per-step visual playback:

1. Export step-local model state plus static plotting inputs.
2. Implement a deterministic chunk renderer for one snapshot at a time.
3. Reuse the same remote mount and chunk merge pattern.

The more the problem can separate immutable inputs from step-local state, the
easier it is to fan render work out across multiple CPUs and machines.
