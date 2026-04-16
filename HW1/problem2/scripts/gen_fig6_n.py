"""Generate Fig. 6-style plot from n-qubit sweep results.

MSE vs system size n (number of qubits) for explicit, reuploading, kernel.
Reproduces the spirit of Jerbi et al. (2023) Fig. 6.

Run locally:
    python HW1/problem2/scripts/gen_fig6_n.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).parents[1] / "fig6_sweep_n" / "results.json"
OUT  = Path(__file__).parents[1] / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads(DATA.read_text())


def stats(records, key):
    by_n = {}
    for r in records:
        n = r["n"]
        by_n.setdefault(n, []).append(r[key])
    ns   = sorted(by_n)
    mean = np.array([np.mean(by_n[n]) for n in ns])
    std  = np.array([np.std( by_n[n]) for n in ns])
    return np.array(ns), mean, std


n_ex, ex_tr_m, ex_tr_s = stats(res["explicit"],    "train_mse")
_,    ex_te_m, ex_te_s = stats(res["explicit"],    "test_mse")
n_re, re_tr_m, re_tr_s = stats(res["reuploading"], "train_mse")
_,    re_te_m, re_te_s = stats(res["reuploading"], "test_mse")
n_ke, ke_tr_m, ke_tr_s = stats(res["kernel"],      "train_mse")
_,    ke_te_m, ke_te_s = stats(res["kernel"],       "test_mse")

fig, ax = plt.subplots(figsize=(7, 4.5))

# Data Reuploading (green) — implicit model in paper
ax.plot(n_re, re_tr_m, "g--x", lw=1.4, ms=7, label="Training reuploading")
ax.fill_between(n_re, re_tr_m - re_tr_s, re_tr_m + re_tr_s, color="green", alpha=0.12)
ax.plot(n_re, re_te_m, "g-D",  lw=1.8, ms=7, label="Testing reuploading")
ax.fill_between(n_re, re_te_m - re_te_s, re_te_m + re_te_s, color="green", alpha=0.15)

# Explicit (red) — explicit model in paper
ax.plot(n_ex, ex_tr_m, "r--x", lw=1.4, ms=7, label="Training explicit")
ax.fill_between(n_ex, ex_tr_m - ex_tr_s, ex_tr_m + ex_tr_s, color="red", alpha=0.12)
ax.plot(n_ex, ex_te_m, "r-D",  lw=1.8, ms=7, label="Testing explicit")
ax.fill_between(n_ex, ex_te_m - ex_te_s, ex_te_m + ex_te_s, color="red", alpha=0.15)

# Implicit Kernel (blue) — classical baseline in paper
ax.plot(n_ke, ke_tr_m, "b--x", lw=1.4, ms=7, label="Training kernel")
ax.fill_between(n_ke, ke_tr_m - ke_tr_s, ke_tr_m + ke_tr_s, color="blue", alpha=0.10)
ax.plot(n_ke, ke_te_m, "b-D",  lw=1.8, ms=7, label="Testing kernel")
ax.fill_between(n_ke, ke_te_m - ke_te_s, ke_te_m + ke_te_s, color="blue", alpha=0.12)

ax.set_xlabel("System size $n$ (number of qubits)", fontsize=12)
ax.set_ylabel("Mean squared error", fontsize=12)
ax.set_xticks(n_ex)
ax.set_ylim(bottom=0)
ax.set_title("Regression performance on hypersphere dataset\n(reproducing Ref. [3] Fig. 6, $n$ = system size)", fontsize=11)
ax.legend(fontsize=8.5, ncol=2, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()

for ext in ["pdf", "png"]:
    path = OUT / f"prob2_a_fig6.{ext}"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved {path}")
plt.close(fig)
