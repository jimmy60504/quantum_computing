"""Fig. 6 sweep — MSE vs system size n (number of qubits) for hypersphere dataset.

Properly reproduces Jerbi et al. (2023) Fig. 6:
- x-axis: n (number of qubits = input dimensionality)
- y-axis: MSE (Brier score)
- 3 methods: explicit, reuploading (≈ implicit), kernel
- 5 seeds for std bands

Run via Docker on gx10 from repo root:
    ./scripts/gx10_run_py.sh HW1/problem2/scripts/fig6_sweep_n.py

Results saved to:
    HW1/problem2/fig6_sweep_n/results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from HW1.problem2.datasets import make_hypersphere_dataset
from HW1.problem2.core.models.explicit    import ExplicitQuantumClassifier
from HW1.problem2.core.models.kernel      import QuantumKernelClassifier
from HW1.problem2.core.models.reuploading import DataReuploadingClassifier

OUT_DIR  = Path(__file__).parents[1] / "fig6_sweep_n"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "results.json"

SEEDS   = [11224001, 42, 137, 2025, 999]
N_VALS  = [2, 3, 4, 5, 6, 7, 8]   # system size (qubits = input dims)
N_LAYERS = 4                        # fixed circuit depth
EPOCHS  = 50
LR      = 0.05
BATCH   = 32
N_SAMP  = 200


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, dtype=float) - labels.astype(float)) ** 2))


results: dict[str, list] = {"explicit": [], "reuploading": [], "kernel": []}
total = len(SEEDS) * len(N_VALS) * 3
done  = 0


def log(tag: str, t0: float, tr: float, te: float) -> None:
    global done
    done += 1
    print(f"[{done:3d}/{total}] {tag:<40s}  tr={tr:.4f}  te={te:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)


for seed in SEEDS:
    for n in N_VALS:
        ds = make_hypersphere_dataset(n_dims=n, n_samples=N_SAMP, random_state=seed)
        Xtr, ytr = ds.X_train, ds.y_train
        Xte, yte = ds.X_test,  ds.y_test

        # explicit
        t0 = time.time()
        em = ExplicitQuantumClassifier(num_layers=N_LAYERS, num_qubits=n, learning_rate=LR, seed=seed)
        for _ in range(EPOCHS):
            em.fit(Xtr, ytr, epochs=1, batch_size=BATCH)
        tr_e = brier(em.forward(Xtr).detach().numpy(), ytr)
        te_e = brier(em.forward(Xte).detach().numpy(), yte)
        log(f"explicit    n={n} seed={seed}", t0, tr_e, te_e)
        results["explicit"].append({"seed": seed, "n": n, "train_mse": tr_e, "test_mse": te_e})

        # reuploading
        t0 = time.time()
        rm = DataReuploadingClassifier(num_layers=N_LAYERS, num_qubits=n, learning_rate=LR, seed=seed)
        for _ in range(EPOCHS):
            rm.fit(Xtr, ytr, epochs=1, batch_size=BATCH)
        tr_r = brier(rm.forward(Xtr).detach().numpy(), ytr)
        te_r = brier(rm.forward(Xte).detach().numpy(), yte)
        log(f"reuploading n={n} seed={seed}", t0, tr_r, te_r)
        results["reuploading"].append({"seed": seed, "n": n, "train_mse": tr_r, "test_mse": te_r})

        # kernel
        t0 = time.time()
        km = QuantumKernelClassifier(num_qubits=n, seed=seed)
        km.fit(Xtr, ytr)
        tr_k = brier(km.decision_function(Xtr), ytr)
        te_k = brier(km.decision_function(Xte), yte)
        log(f"kernel      n={n} seed={seed}", t0, tr_k, te_k)
        results["kernel"].append({"seed": seed, "n": n, "train_mse": tr_k, "test_mse": te_k})

        OUT_FILE.write_text(json.dumps(results, indent=2))

print(f"\nDone. Results saved to {OUT_FILE}")
