"""
preview_prob2.py

Visualize Problem 2 datasets and sketch the expected UI layout:
  - Circle dataset (same as Ref. [4])
  - Moons dataset (sklearn, noise=0.1, n_samples=200)

Outputs a 3x2 grid of decision-boundary mockups using simple classical
classifiers as placeholders (SVM-RBF for all three slots), plus a standalone
dataset overview figure.

Run with:
    bash scripts/gx10_run_py.sh HW1/problem2/preview.py

Outputs go to hf_space_hw1_problem2/assets/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

SEED = 11224001  # replace with actual student ID numerical part

rng = np.random.default_rng(SEED)

# ── datasets ──────────────────────────────────────────────────────────────────

def make_circle_dataset(n=200, seed=SEED):
    """PennyLane demo circle dataset: points in [-1, 1]^2 labeled by radius."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(low=-1.0, high=1.0, size=(n, 2))
    radius = np.sqrt(2 / np.pi)
    y = (np.linalg.norm(X, axis=1) < radius).astype(int)
    return X, y


def make_moons_dataset(n=200, seed=SEED):
    X, y = make_moons(n_samples=n, noise=0.1, random_state=seed)
    return X, y


# ── plotting helpers ───────────────────────────────────────────────────────────

CMAP_BG = ListedColormap(["#d0e8ff", "#ffd0d0"])
COLORS   = ["#1565c0", "#c62828"]
MARKERS  = ["o", "s"]

METHOD_LABELS = [
    "Explicit Quantum Model\n(placeholder: SVM-RBF)",
    "Implicit Quantum Kernel\n(placeholder: SVM-RBF)",
    "Data Reuploading\n(placeholder: SVM-RBF)",
]
DATASET_LABELS = ["Circle dataset", "Moons dataset"]


def plot_decision_boundary(ax, clf, X, y, title, h=0.02):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    x_min, x_max = Xs[:, 0].min() - 0.5, Xs[:, 0].max() + 0.5
    y_min, y_max = Xs[:, 1].min() - 0.5, Xs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf.fit(Xs, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=CMAP_BG, alpha=0.6)
    ax.contour(xx, yy, Z, colors="white", linewidths=0.8, alpha=0.5)

    for cls, color, marker in zip([0, 1], COLORS, MARKERS):
        mask = y == cls
        ax.scatter(Xs[mask, 0], Xs[mask, 1],
                   c=color, marker=marker, s=20, edgecolors="white",
                   linewidths=0.4, label=f"Class {cls}", zorder=3)

    acc = (clf.predict(Xs) == y).mean()
    ax.set_title(f"{title}\n(placeholder acc: {acc:.2f})", fontsize=8, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


# ── figure 1: dataset overview ─────────────────────────────────────────────────

X_circ, y_circ   = make_circle_dataset()
X_moons, y_moons = make_moons_dataset()

fig0, axes0 = plt.subplots(1, 2, figsize=(8, 3.5))
fig0.suptitle("Problem 2 — Dataset Overview", fontsize=13, fontweight="bold")

for ax, X, y, title in zip(axes0,
                            [X_circ, X_moons],
                            [y_circ, y_moons],
                            DATASET_LABELS):
    Xs = StandardScaler().fit_transform(X)
    for cls, color, marker in zip([0, 1], COLORS, MARKERS):
        mask = y == cls
        ax.scatter(Xs[mask, 0], Xs[mask, 1],
                   c=color, marker=marker, s=25, edgecolors="white",
                   linewidths=0.4, label=f"Class {cls}", zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")

fig0.tight_layout()
out0 = "HW1/problem2/hf_space/assets/preview_datasets.png"
fig0.savefig(out0, dpi=150, bbox_inches="tight")
print(f"Saved: {out0}")


# ── figure 2: 3×2 decision-boundary grid (placeholder) ────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(8, 11))
fig.suptitle(
    "Problem 2 — Decision Boundaries (3 methods × 2 datasets)\n"
    "[placeholder: SVM-RBF; will be replaced with real QML results]",
    fontsize=11, fontweight="bold"
)

# column headers
for ax, ds_label in zip(axes[0], DATASET_LABELS):
    ax.set_title(ds_label, fontsize=10, fontweight="bold", color="#333")

datasets = [(X_circ, y_circ), (X_moons, y_moons)]

for row, method_label in enumerate(METHOD_LABELS):
    for col, (X, y) in enumerate(datasets):
        ax = axes[row][col]
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        subtitle = method_label.split("\n")[0]
        plot_decision_boundary(ax, clf, X, y, subtitle)
        if col == 0:
            ax.set_ylabel(method_label, fontsize=8, labelpad=6)

# shared legend
handles = [
    mpatches.Patch(color=COLORS[0], label="Class 0"),
    mpatches.Patch(color=COLORS[1], label="Class 1"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, 0.0))

fig.tight_layout(rect=[0, 0.03, 1, 1])
out1 = "HW1/problem2/hf_space/assets/preview_boundaries.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")


# ── figure 3: UI wireframe sketch ─────────────────────────────────────────────
# Shows how the web UI sections might be arranged

fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6))
fig2.patch.set_facecolor("#1a1a2e")

titles = [
    "Explicit — Circle",  "Implicit Kernel — Circle",  "Data Reuploading — Circle",
    "Explicit — Moons",   "Implicit Kernel — Moons",   "Data Reuploading — Moons",
]
XY = [(X_circ, y_circ)] * 3 + [(X_moons, y_moons)] * 3

for ax, (X, y), title in zip(axes2.flat, XY, titles):
    ax.set_facecolor("#0f3460")
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    h = 0.04
    x_min, x_max = Xs[:, 0].min() - 0.4, Xs[:, 0].max() + 0.4
    y_min, y_max = Xs[:, 1].min() - 0.4, Xs[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    clf.fit(Xs, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z,
                colors=["#16213e", "#e94560"], alpha=0.5)
    ax.contour(xx, yy, Z, colors="white", linewidths=0.6, alpha=0.7)
    for cls, color, marker in zip([0, 1], ["#4fc3f7", "#ef9a9a"], MARKERS):
        mask = y == cls
        ax.scatter(Xs[mask, 0], Xs[mask, 1],
                   c=color, marker=marker, s=18, edgecolors="none", zorder=3)
    acc = (clf.predict(Xs) == y).mean()
    ax.set_title(title, fontsize=9, color="white", pad=4)
    ax.text(0.97, 0.04, f"acc {acc:.2f}", transform=ax.transAxes,
            color="#aef", fontsize=7, ha="right")
    ax.set_xticks([])
    ax.set_yticks([])

fig2.suptitle("Problem 2 — QML Classifier Explorer  [UI dark theme sketch]",
              fontsize=12, color="white", fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
out2 = "HW1/problem2/hf_space/assets/preview_ui_sketch.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved: {out2}")

print("\nDone. Three preview images written to scripts/:")
print(f"  {out0}")
print(f"  {out1}")
print(f"  {out2}")
