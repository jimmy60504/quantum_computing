# Viewer Access — HW1 Problem 2 QML Viewer

The static viewer lives in `HW1/problem2/hf_space/` on the repo.
Runtime data (large JSON exports) live on **gx10** only and should never be
synced back to local.

## How to start the viewer

### 1. Start HTTP server on gx10 (if not already running)

```bash
ssh gx10 "tmux new-session -d -s prob2_viewer \
  'cd /home/jimmy/quantum_computing/HW1/problem2/hf_space && python3 -m http.server 8787'"
```

Check if already running:
```bash
ssh gx10 "tmux ls 2>/dev/null | grep prob2_viewer || echo 'not running'"
```

### 2. Open SSH tunnel from local

```bash
ssh -f -N -L 8787:localhost:8787 gx10
```

### 3. Open in browser

```
http://localhost:8787
```

## Runtime data location on gx10

```
/home/jimmy/quantum_computing/HW1/problem2/hf_space/runtime/
  viewer_manifest.json          ← manifest listing all runs
  q2-le4-lr4-e50-n200.json     ← latest exp1 run (assemble output)
  q2-l4-e30.json               ← earlier run
  ...
```

Artifact JSONs (pre-assemble) live at:
```
/home/jimmy/quantum_computing/logs/hw1_prob2_pipeline/runs/exp1/
  datasets.npz
  explicit_artifact.json
  reuploading_artifact.json
  kernel_artifact.json
```

## Re-running the pipeline

Use the staged script already on gx10:
```bash
ssh gx10 "tmux new-session -d -s prob2_pipeline \
  'bash /home/jimmy/quantum_computing/logs/hw1_prob2_staged_run.sh \
   2>&1 | tee /home/jimmy/quantum_computing/logs/hw1_prob2_staged_run.log'"
```

Monitor:
```bash
ssh gx10 "tail -f /home/jimmy/quantum_computing/logs/hw1_prob2_staged_run.log"
```

## Closing the tunnel

```bash
pkill -f "ssh -f -N -L 8787"
```
