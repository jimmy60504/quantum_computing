#!/usr/bin/env python3
"""Generate training-curve figure for HW1 Problem 3(d).

Reads the runtime JSON and produces a 2-panel plot:
  Left:  Loss vs. Epoch   (train & test, both models)
  Right: Accuracy vs. Epoch (train & test, both models)

Outputs PDF + PNG into HW1/problem3/report_figs/.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "hf_space" / "runtime" / "qnn-8q4l-vmap.json"
OUT  = ROOT / "report_figs"
OUT.mkdir(exist_ok=True)

with open(DATA) as f:
    data = json.load(f)

# ── extract histories ──────────────────────────────────────────────
def extract(method_key):
    hist = data["methods"][method_key]["history"]
    return {
        "epoch":      np.array([h["epoch"] for h in hist]),
        "train_loss": np.array([h["train_loss"] for h in hist]),
        "test_loss":  np.array([h["test_loss"] for h in hist]),
        "train_acc":  np.array([h["train_acc"] for h in hist]) * 100,
        "test_acc":   np.array([h["test_acc"] for h in hist]) * 100,
    }

mlp = extract("mlp")
qnn = extract("qnn")

# ── plot ───────────────────────────────────────────────────────────
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

# colours
C_MLP = "#1f77b4"
C_QNN = "#d62728"

# --- Loss ---
ax_loss.plot(mlp["epoch"], mlp["train_loss"], "-o",  color=C_MLP, ms=4, label="MLP train")
ax_loss.plot(mlp["epoch"], mlp["test_loss"],  "--s", color=C_MLP, ms=4, label="MLP test")
ax_loss.plot(qnn["epoch"], qnn["train_loss"], "-o",  color=C_QNN, ms=4, label="QNN train")
ax_loss.plot(qnn["epoch"], qnn["test_loss"],  "--s", color=C_QNN, ms=4, label="QNN test")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Cross-Entropy Loss")
ax_loss.set_title("Loss")
ax_loss.legend(fontsize=8)
ax_loss.set_xlim(0.5, 20.5)
ax_loss.grid(alpha=0.3)

# --- Accuracy ---
ax_acc.plot(mlp["epoch"], mlp["train_acc"], "-o",  color=C_MLP, ms=4, label="MLP train")
ax_acc.plot(mlp["epoch"], mlp["test_acc"],  "--s", color=C_MLP, ms=4, label="MLP test")
ax_acc.plot(qnn["epoch"], qnn["train_acc"], "-o",  color=C_QNN, ms=4, label="QNN train")
ax_acc.plot(qnn["epoch"], qnn["test_acc"],  "--s", color=C_QNN, ms=4, label="QNN test")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_title("Accuracy")
ax_acc.legend(fontsize=8)
ax_acc.set_xlim(0.5, 20.5)
ax_acc.grid(alpha=0.3)

fig.suptitle("Problem 3(d): CNN+MLP vs CNN+QNN Training Curves (CIFAR-10, 20 Epochs)",
             fontsize=11, y=1.02)
fig.tight_layout()

for ext in ("pdf", "png"):
    out_path = OUT / f"prob3_d_training_curves.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
