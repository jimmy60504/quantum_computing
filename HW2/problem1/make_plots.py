"""HW2 Problem 1 — generate figures for the report.

Runs the experiments and writes figures to HW2/problem1/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import time

from solution import (
    CAPACITY, N_ITEMS, VALUES, WEIGHTS,
    build_qubo, solve_brute_force, solve_exact, sweep_num_reads,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

LAMBDAS = [0.1, 0.3, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0]
NUM_READS_LIST = [10, 100, 1000, 10000]
SA_LAMBDA = 8.0


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_all() -> tuple[dict, float]:
    t0 = time.perf_counter()
    classical = solve_brute_force()
    classical_time = time.perf_counter() - t0
    optimal_bits = [1 if i in classical.selected else 0 for i in range(N_ITEMS)]

    lam_rows = []
    exact_time_at_sa_lambda = 0.0
    for lam in LAMBDAS:
        Q = build_qubo(lam)
        t0 = time.perf_counter()
        sample, energy = solve_exact(Q)
        elapsed = time.perf_counter() - t0
        if lam == SA_LAMBDA:
            exact_time_at_sa_lambda = elapsed
        x = np.array([sample[i] for i in range(N_ITEMS)])
        weight = int(WEIGHTS @ x)
        value  = int(VALUES  @ x)
        lam_rows.append(dict(
            lam=lam, value=value, weight=weight, energy=float(energy),
            feasible=weight <= CAPACITY,
            optimal=(weight <= CAPACITY and value == classical.total_value),
        ))

    Q_sa = build_qubo(SA_LAMBDA)
    sa_rows = sweep_num_reads(NUM_READS_LIST, Q_sa, optimal_bits)

    return dict(
        classical=dict(
            selected=classical.selected,
            total_weight=classical.total_weight,
            total_value=classical.total_value,
        ),
        classical_time=classical_time,
        lam_rows=lam_rows,
        sa_rows=sa_rows,
    ), exact_time_at_sa_lambda


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_lambda_sweep(data: dict) -> None:
    rows = data["lam_rows"]
    opt_value = data["classical"]["total_value"]

    lams = [r["lam"] for r in rows]
    values = [r["value"] for r in rows]
    feas = [r["feasible"] for r in rows]
    weights = [r["weight"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = ["tab:blue" if f else "tab:red" for f in feas]
    ax1.scatter(lams, values, c=colors, s=80, zorder=3)
    ax1.axhline(opt_value, ls="--", c="k", lw=1, label=f"optimum = {opt_value}")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("ExactSolver value")
    ax1.set_title("Recovered value vs penalty")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(lams, weights, c=colors, s=80, zorder=3)
    ax2.axhline(CAPACITY, ls="--", c="k", lw=1, label=f"capacity = {CAPACITY}")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("ExactSolver weight")
    ax2.set_title("Recovered weight vs penalty")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Part (b): ExactSolver QUBO under penalty sweep "
                 "(blue = feasible, red = infeasible)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_lambda_sweep.png", dpi=140)
    plt.close(fig)


def fig_sa_success(data: dict) -> None:
    rows = data["sa_rows"]
    n_reads = np.array([r["num_reads"] for r in rows])
    p = np.array([r["success_prob"] for r in rows])
    succ = (p * n_reads).astype(int)

    # Wilson 95% CI
    z = 1.96
    denom = 1 + z**2 / n_reads
    centre = (p + z**2 / (2 * n_reads)) / denom
    half = z * np.sqrt(p * (1 - p) / n_reads + z**2 / (4 * n_reads**2)) / denom
    lo, hi = np.clip(centre - half, 1e-6, 1), np.clip(centre + half, 1e-6, 1)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(n_reads, np.clip(p, 1e-6, 1), yerr=[np.clip(p, 1e-6, 1) - lo, hi - np.clip(p, 1e-6, 1)],
                fmt="o-", capsize=4, lw=2, ms=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("num_reads")
    ax.set_ylabel("success probability (Wilson 95% CI)")
    ax.set_title(f"Part (c): SA hit-rate on optimum (λ={SA_LAMBDA})")
    ax.grid(True, which="both", alpha=0.3)
    for nr, pr, k in zip(n_reads, p, succ):
        ax.annotate(f"{k}/{nr}", (nr, max(pr, 1e-6)), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_sa_success.png", dpi=140)
    plt.close(fig)


def fig_energy_hist(data: dict) -> None:
    # Use the largest num_reads run for a meaningful distribution
    largest = max(data["sa_rows"], key=lambda r: r["num_reads"])
    values = np.array(largest["sample_values"])
    opt = data["classical"]["total_value"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(values, bins=40, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(opt, color="tab:red", ls="--", lw=2, label=f"optimum = {opt}")
    ax.set_xlabel("Sample value (Σ v·x)")
    ax.set_ylabel("Count")
    ax.set_title(f"Part (c): SA value distribution at num_reads={largest['num_reads']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_energy_hist.png", dpi=140)
    plt.close(fig)


def fig_comparison(data: dict, exact_time: float) -> None:
    rows = data["sa_rows"]
    classical = data["classical"]
    opt = classical["total_value"]

    methods = ["Classical\n(brute force)", f"Exact QUBO\n(λ={SA_LAMBDA})"] + [
        f"SA\nnr={r['num_reads']}" for r in rows
    ]
    # Mean sample value better separates the SA configs (best-of-N hides info)
    sa_means = [float(np.mean(r["sample_values"])) for r in rows]
    means = [opt, opt] + sa_means
    bests = [opt, opt] + [r["best_value"] for r in rows]
    times = [data.get("classical_time", 0.001), exact_time] + [r["time_s"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(methods))
    width = 0.4
    ax1.bar(x - width/2, bests, width, color="tab:blue", alpha=0.85, label="best value")
    ax1.bar(x + width/2, means, width, color="tab:cyan", alpha=0.85, label="mean value")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("Value")
    ax1.axhline(opt, ls="--", c="k", lw=1, alpha=0.5)
    ax1.set_title("Part (d): Method comparison")
    ax1.legend(loc="lower left")
    for xi, b, m in zip(x, bests, means):
        ax1.text(xi - width/2, b + 4, f"{int(b)}", ha="center", fontsize=8)
        ax1.text(xi + width/2, m + 4, f"{m:.0f}", ha="center", fontsize=8, color="tab:cyan")

    ax2 = ax1.twinx()
    ax2.plot(x, times, "o-", color="tab:red", lw=1.5, ms=7)
    ax2.set_ylabel("Time (s)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_comparison.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data, exact_time = run_all()
    (FIG_DIR / "results.json").write_text(json.dumps(data, indent=2, default=float))

    fig_lambda_sweep(data)
    fig_sa_success(data)
    fig_energy_hist(data)
    fig_comparison(data, exact_time)
    print(f"wrote 4 figures + results.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
