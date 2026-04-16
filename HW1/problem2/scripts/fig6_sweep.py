"""Fig. 6 sweep — MSE vs number of layers (L) for circle dataset.

Reproduces the spirit of Jerbi et al. (2023) Fig. 6, replacing
"system size n" with "number of variational layers L".

Run via Docker on gx10 from repo root:
    ./scripts/gx10_run_py.sh HW1/problem2/scripts/fig6_sweep.py

Results saved to:
    HW1/problem2/fig6_sweep/results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── imports (works inside Docker /workspace) ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root

from HW1.problem2.datasets import make_circle_dataset
from HW1.problem2.core.models.explicit    import ExplicitQuantumClassifier
from HW1.problem2.core.models.kernel      import QuantumKernelClassifier
from HW1.problem2.core.models.reuploading import DataReuploadingClassifier

# ── config ────────────────────────────────────────────────────────────────────
SEEDS   = [11224001, 42, 137, 2025, 999]
L_VALS  = [1, 2, 3, 4, 5, 6, 7, 8]
EPOCHS  = 50
LR      = 0.05
BATCH   = 32
N_SAMP  = 200

OUT_DIR = Path(__file__).parents[1] / "fig6_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "results.json"


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, dtype=float) - labels.astype(float)) ** 2))


def run_explicit(L: int, seed: int, Xtr, ytr, Xte, yte):
    m = ExplicitQuantumClassifier(
        num_layers=L, num_qubits=2, learning_rate=LR, seed=seed)
    for _ in range(EPOCHS):
        m.fit(Xtr, ytr, epochs=1, batch_size=BATCH)
    return (brier(m.forward(Xtr).detach().numpy(), ytr),
            brier(m.forward(Xte).detach().numpy(), yte))


def run_reuploading(L: int, seed: int, Xtr, ytr, Xte, yte):
    m = DataReuploadingClassifier(
        num_layers=L, num_qubits=2, learning_rate=LR, seed=seed)
    for _ in range(EPOCHS):
        m.fit(Xtr, ytr, epochs=1, batch_size=BATCH)
    return (brier(m.forward(Xtr).detach().numpy(), ytr),
            brier(m.forward(Xte).detach().numpy(), yte))


def run_kernel(seed: int, Xtr, ytr, Xte, yte):
    m = QuantumKernelClassifier(num_qubits=2, seed=seed)
    m.fit(Xtr, ytr)
    return (brier(m.decision_function(Xtr), ytr),
            brier(m.decision_function(Xte), yte))


# ── sweep ─────────────────────────────────────────────────────────────────────
results: dict[str, list] = {"explicit": [], "reuploading": [], "kernel": []}
total = len(SEEDS) * (2 * len(L_VALS) + 1)
done  = 0


def log(tag: str, t0: float, tr: float, te: float):
    global done
    done += 1
    print(f"[{done:3d}/{total}] {tag:<35s}  tr={tr:.4f}  te={te:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)


for seed in SEEDS:
    ds = make_circle_dataset(n_samples=N_SAMP, random_state=seed)
    Xtr, ytr = ds.X_train, ds.y_train
    Xte, yte = ds.X_test,  ds.y_test

    # kernel — L-independent
    t0 = time.time()
    tr_k, te_k = run_kernel(seed, Xtr, ytr, Xte, yte)
    log(f"kernel  seed={seed}", t0, tr_k, te_k)
    results["kernel"].append({"seed": seed, "L": None, "train_mse": tr_k, "test_mse": te_k})
    OUT_FILE.write_text(json.dumps(results, indent=2))

    for L in L_VALS:
        t0 = time.time()
        tr_e, te_e = run_explicit(L, seed, Xtr, ytr, Xte, yte)
        log(f"explicit  L={L}  seed={seed}", t0, tr_e, te_e)
        results["explicit"].append({"seed": seed, "L": L, "train_mse": tr_e, "test_mse": te_e})

        t0 = time.time()
        tr_r, te_r = run_reuploading(L, seed, Xtr, ytr, Xte, yte)
        log(f"reuploading  L={L}  seed={seed}", t0, tr_r, te_r)
        results["reuploading"].append({"seed": seed, "L": L, "train_mse": tr_r, "test_mse": te_r})

        OUT_FILE.write_text(json.dumps(results, indent=2))

print(f"\nDone. Results saved to {OUT_FILE}")
