import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/workspace/HW1/problem2/fig6_sweep/results.json")
OUT  = Path("/workspace/HW1/problem2/report_figs")
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads(DATA.read_text())

def stats(records, key):
    by_L = {}
    for r in records:
        L = r["L"]
        by_L.setdefault(L, []).append(r[key])
    Ls   = sorted(k for k in by_L if k is not None)
    mean = np.array([np.mean(by_L[L]) for L in Ls])
    std  = np.array([np.std( by_L[L]) for L in Ls])
    return np.array(Ls), mean, std

k_tr_vals = [r["train_mse"] for r in res["kernel"]]
k_te_vals = [r["test_mse"]  for r in res["kernel"]]
k_tr_mean, k_tr_std = np.mean(k_tr_vals), np.std(k_tr_vals)
k_te_mean, k_te_std = np.mean(k_te_vals), np.std(k_te_vals)

L_ex, ex_tr_m, ex_tr_s = stats(res["explicit"],    "train_mse")
_,    ex_te_m, ex_te_s = stats(res["explicit"],    "test_mse")
L_re, re_tr_m, re_tr_s = stats(res["reuploading"], "train_mse")
_,    re_te_m, re_te_s = stats(res["reuploading"], "test_mse")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(L_re, re_tr_m, "g--x", lw=1.4, ms=7, label="Training reuploading")
ax.fill_between(L_re, re_tr_m - re_tr_s, re_tr_m + re_tr_s, color="green", alpha=0.12)
ax.plot(L_re, re_te_m, "g-D",  lw=1.8, ms=7, label="Testing reuploading")
ax.fill_between(L_re, re_te_m - re_te_s, re_te_m + re_te_s, color="green", alpha=0.15)

ax.plot(L_ex, ex_tr_m, "r--x", lw=1.4, ms=7, label="Training explicit")
ax.fill_between(L_ex, ex_tr_m - ex_tr_s, ex_tr_m + ex_tr_s, color="red", alpha=0.12)
ax.plot(L_ex, ex_te_m, "r-D",  lw=1.8, ms=7, label="Testing explicit")
ax.fill_between(L_ex, ex_te_m - ex_te_s, ex_te_m + ex_te_s, color="red", alpha=0.15)

L_range = np.array([L_ex[0], L_ex[-1]])
ax.plot(L_range, [k_tr_mean]*2, "b--x", lw=1.4, ms=7, label="Training kernel")
ax.fill_between(L_range, k_tr_mean - k_tr_std, k_tr_mean + k_tr_std, color="blue", alpha=0.10)
ax.plot(L_range, [k_te_mean]*2, "b-D",  lw=1.8, ms=7, label="Testing kernel")
ax.fill_between(L_range, k_te_mean - k_te_std, k_te_mean + k_te_std, color="blue", alpha=0.12)

ax.set_xlabel("Number of layers ($L$)", fontsize=12)
ax.set_ylabel("Mean squared error", fontsize=12)
ax.set_xticks(L_ex)
ax.set_ylim(bottom=0)
ax.set_title("Regression performance on circle dataset\n(reproducing Ref. [3] Fig. 6 style)", fontsize=11)
ax.legend(fontsize=8.5, ncol=2, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()

for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"prob2_a_fig6.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved")
