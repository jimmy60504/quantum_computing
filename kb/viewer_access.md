# Viewer Access — HW1 Static Viewers

Each problem has its own static viewer under `HW1/problemX/hf_space/` on gx10.
Runtime data (large JSON exports) live on **gx10** only and should never be
synced back to local.

## Port assignments

| Problem | tmux session     | gx10 port | local URL                   |
|---------|------------------|-----------|-----------------------------|
| prob1   | `prob1_viewer`   | 8781      | http://localhost:8781       |
| prob2   | `prob2_viewer`   | 8782      | http://localhost:8782       |
| prob3   | `prob3_viewer`   | 8783      | http://localhost:8783       |

## Start a viewer on gx10

Replace `probX` / `878X` / `problemX` with the values from the table above.

```bash
# Check if already running
ssh gx10 "tmux ls 2>/dev/null | grep probX_viewer || echo 'not running'"

# Start (if not running)
ssh gx10 "tmux new-session -d -s probX_viewer \
  'cd /home/jimmy/quantum_computing/HW1/problemX/hf_space && python3 -m http.server 878X'"
```

Quick shortcuts for each problem:

```bash
# prob1
ssh gx10 "tmux new-session -d -s prob1_viewer \
  'cd /home/jimmy/quantum_computing/HW1/problem1/hf_space && python3 -m http.server 8781'"

# prob2
ssh gx10 "tmux new-session -d -s prob2_viewer \
  'cd /home/jimmy/quantum_computing/HW1/problem2/hf_space && python3 -m http.server 8782'"

# prob3
ssh gx10 "tmux new-session -d -s prob3_viewer \
  'cd /home/jimmy/quantum_computing/HW1/problem3/hf_space && python3 -m http.server 8783'"
```

## Open SSH tunnels from local

Open all three at once:
```bash
ssh -f -N -L 8781:localhost:8781 -L 8782:localhost:8782 -L 8783:localhost:8783 gx10
```

Or one at a time:
```bash
ssh -f -N -L 8781:localhost:8781 gx10   # prob1
ssh -f -N -L 8782:localhost:8782 gx10   # prob2
ssh -f -N -L 8783:localhost:8783 gx10   # prob3
```

## Close tunnels

```bash
pkill -f "ssh -f -N -L 878"
```

---

## Runtime data locations on gx10

### Problem 1
```
/home/jimmy/quantum_computing/HW1/problem1/hf_space/runtime/
  viewer_manifest.json
  ...
```

### Problem 2
```
/home/jimmy/quantum_computing/HW1/problem2/hf_space/runtime/
  viewer_manifest.json
  q2-le4-lr4-e50-n200.json
  q2-l4-e30.json
  ...
```

Artifact JSONs (pre-assemble):
```
/home/jimmy/quantum_computing/logs/hw1_prob2_pipeline/runs/exp1/
  datasets.npz
  explicit_artifact.json
  reuploading_artifact.json
  kernel_artifact.json
```

### Problem 3
```
/home/jimmy/quantum_computing/HW1/problem3/hf_space/runtime/
  viewer_manifest.json
  ...
```

## Re-running the prob2 pipeline

```bash
ssh gx10 "tmux new-session -d -s prob2_pipeline \
  'bash /home/jimmy/quantum_computing/logs/hw1_prob2_staged_run.sh \
   2>&1 | tee /home/jimmy/quantum_computing/logs/hw1_prob2_staged_run.log'"
```

Monitor:
```bash
ssh gx10 "tail -f /home/jimmy/quantum_computing/logs/hw1_prob2_staged_run.log"
```
