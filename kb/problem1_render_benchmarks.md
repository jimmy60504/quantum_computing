# Problem 1 Render Benchmarks

This note records the render benchmarks collected while tuning the distributed
viewer workflow for `HW1/problem1`.

## Scope

- Workload: render snapshot chunks for `raw-q2-l2-e20-dist`
- Snapshot export:
  `hf_space_hw1_problem1/runtime/raw-q2-l2-e20-dist_snapshots.json`
- Main short benchmark slice: `0:32`
- Main long benchmark slice: `0:224`

## Baseline observations

- The first slow `gx10` runs were not limited by ARM alone.
- The biggest regressions came from:
  - rebuilding datasets and target heatmaps for every snapshot
  - using `spawn` on Linux workers
  - letting each render worker also open its own BLAS/OpenMP thread pool
- After fixing those three issues, `gx10` throughput improved dramatically.

## Code changes that affected the results

- `environment.yml`
  - added `pytorch`
- `HW1/problem1/core/modeling.py`
  - cache train/test tensors per worker
  - cache fixed target heatmaps per worker
  - default Linux render multiprocessing to `fork`
  - add a configurable `imap` chunksize
- `HW1/problem1/tools/render_snapshot_chunk.py`
  - force per-worker math-library thread counts to `1`

## Cross-machine comparison

These numbers are the most useful reference points gathered so far.

| Machine | Environment | Workers | Slice | Time |
| --- | --- | ---: | --- | ---: |
| `macbook` (`Apple M5`) | native Conda | 4 | `0:32` | `29.94s` |
| local `Mac Studio` (`M1 Max`) | native venv | 6 | `0:32` | `50.13s` |
| `deck` (`x86_64`) | Docker render worker | 4 | `224:320` | about `6m19s` |
| `deck` (`x86_64`) | Docker render worker | 6 | `0:224` | about `12m51s` |
| `gx10` old path | Docker, before render fixes | 6 | `0:32` | about `11m05s` |
| `gx10` tuned path | Docker, after render fixes | 6 | `0:32` | `22.07s` |

Notes:

- The two `deck` numbers were observed from the live run output and chunk
  completion timing. Their log files captured progress but not a final `real`
  line.
- The `Mac Studio` native run completed at `real 50.13` in the terminal. The
  final log file only keeps the render summary payload.
- The `macbook` native run completed at `real 29.94` in the terminal. The same
  number should be treated as the benchmark result even if the remote log
  capture is sparse.

## Normalized comparison

The raw benchmark slices are not all the same length, so the table below
normalizes them into:

- `sec/step`
- `steps/s`
- projected time for `224` steps

This is the easiest way to compare machines on one practical scale.

| Machine | Reference run | sec/step | steps/s | projected `224` steps |
| --- | --- | ---: | ---: | ---: |
| `gx10` tuned, all-core, `workers=20` | `224 steps in 61.08s` | `0.273s` | `3.67` | `61.08s` |
| `macbook` native (`Apple M5`) | `32 steps in 29.94s` | `0.936s` | `1.07` | `209.58s` |
| `gx10` tuned, big-core only, `workers=6` | `32 steps in 22.07s` | `0.690s` | `1.45` | `154.49s` |
| local `Mac Studio` native (`M1 Max`) | `32 steps in 50.13s` | `1.567s` | `0.64` | `350.91s` |
| `deck` Docker render worker | `224 steps in about 771s` | `3.442s` | `0.29` | `771.00s` |
| `gx10` old path before render fixes | `32 steps in about 665s` | `20.781s` | `0.05` | `4655.00s` |

Two practical readings from the normalized view:

- The tuned `gx10` path is now the fastest option we have measured for this
  render workload.
- Native Apple Silicon is still very strong, but the tuned `gx10` all-core
  render path is now clearly ahead on sustained throughput.

## Relative speedups

Using the normalized `sec/step` comparison:

- tuned `gx10` all-core (`workers=20`) is about `3.4x` faster than native
  `macbook`
- tuned `gx10` all-core (`workers=20`) is about `5.7x` faster than native
  local `Mac Studio`
- tuned `gx10` all-core (`workers=20`) is about `12.6x` faster than `deck`
- tuned `gx10` all-core (`workers=20`) is about `76.2x` faster than the old
  unfixed `gx10` render path

## `gx10` short-slice scaling

Source:

- `logs/benchmarks/gx10_render_scaling/summary.tsv`

All runs below used:

- `PROB1_RENDER_MP_START=fork`
- `PROB1_RENDER_WORKER_THREADS=1`
- snapshot slice `0:32`

| Mode | CPU set | CPUs | Workers | Time |
| --- | --- | ---: | ---: | ---: |
| big only | `5-9,15-19` | 10 | 6 | `22.07s` |
| big only | `5-9,15-19` | 10 | 8 | `15.82s` |
| big only | `5-9,15-19` | 10 | 10 | `15.57s` |
| all cores | `0-19` | 20 | 10 | `15.59s` |
| all cores | `0-19` | 20 | 12 | `12.71s` |
| all cores | `0-19` | 20 | 14 | `12.65s` |
| all cores | `0-19` | 20 | 20 | `12.49s` |

Takeaway:

- After the render fixes, using all cores beats staying on big cores only.
- For the short slice, `workers=20` was the fastest measured point, but the
  gap between `14` and `20` was small.

## `gx10` long-slice comparison

Source:

- `logs/benchmarks/gx10_render_scaling/long-compare-summary.tsv`

All runs below used:

- `GX10_CPUSET=0-19`
- `GX10_CPUS=20`
- `PROB1_RENDER_MP_START=fork`
- `PROB1_RENDER_WORKER_THREADS=1`
- snapshot slice `0:224`

| Workers | Time |
| ---: | ---: |
| 14 | `70.62s` |
| 20 | `61.08s` |

Takeaway:

- On the longer slice, `workers=20` was not just slightly faster.
- `workers=20` beat `workers=14` by about `9.54s`, or about `13.5%`.
- The longer run therefore supports using all cores and all workers on `gx10`
  for this specific render workload.

## Current recommendation

For `gx10` Problem 1 snapshot rendering, the current best-known settings are:

```bash
PROB1_RENDER_MP_START=fork \
PROB1_RENDER_WORKER_THREADS=1 \
GX10_CPUSET=0-19 \
GX10_CPUS=20
```

and:

```bash
--render-workers 20
```

If a lighter setting is needed to leave more room for other tasks on `gx10`,
the next most reasonable fallback is:

```bash
GX10_CPUSET=0-19
GX10_CPUS=20
--render-workers 14
```

## Open questions

- The current best setting is tuned for Problem 1 render throughput, not for
  mixed training plus render concurrency.
- `macbook` showed an OpenMP duplicate-runtime warning that was bypassed with
  `KMP_DUPLICATE_LIB_OK=TRUE`. That machine is fast, but its local environment
  may still benefit from a cleaner OpenMP setup.
- It may still be worth testing `imap_unordered` plus final step sorting for
  heterogeneous-core hosts, though the current gains already solved the main
  `gx10` bottleneck.
