"""HW2 Problem 2 — generate figures for the report.

Figures:
  fig1_graph.png         — random graph with optimal Max-Cut partition coloured
  fig2_landscape.png     — F(γ,β) heatmap at p=1 with global minimum marked
  fig3_qaoa_convergence  — ⟨H_C⟩ vs Adam step for p ∈ {1,2,3,4}
  fig4_comparison.png    — approximation ratio bar chart for all methods
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from solution import (
    SEED, build_graph, brute_force_maxcut, cut_value,
    landscape_p1, run_qaoa, solve_sa,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

P_LIST = [1, 2, 3, 4]
N_GRID = 60
N_STEPS = 200
NUM_READS = 1000


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_all() -> dict:
    G = build_graph(SEED)
    out: dict = dict(
        graph=dict(nodes=list(G.nodes), edges=list(G.edges)),
        seed=SEED,
    )

    t0 = time.perf_counter()
    classical = brute_force_maxcut(G)
    out["bruteforce"] = dict(
        bitstring=classical.bitstring,
        cut=classical.cut_value,
        partitions=classical.partitions,
        time_s=time.perf_counter() - t0,
    )

    print(f"  landscape ({N_GRID}×{N_GRID})...")
    t0 = time.perf_counter()
    land = landscape_p1(G, n_grid=N_GRID)
    out["landscape"] = dict(
        gammas=land["gammas"].tolist(),
        betas=land["betas"].tolist(),
        energy=land["energy"].tolist(),
        gamma_min=land["gamma_min"],
        beta_min=land["beta_min"],
        energy_min=land["energy_min"],
        time_s=time.perf_counter() - t0,
    )

    out["qaoa"] = []
    for p in P_LIST:
        print(f"  QAOA p={p}...")
        t0 = time.perf_counter()
        row = run_qaoa(G, p=p, n_steps=N_STEPS)
        elapsed = time.perf_counter() - t0
        out["qaoa"].append(dict(
            p=p,
            cut=row["cut"],
            ratio=row["cut"] / classical.cut_value,
            bitstring=row["bitstring"],
            energies=row["energies"],
            params=row["params_opt"].tolist(),
            time_s=elapsed,
        ))

    print("  SA...")
    t0 = time.perf_counter()
    sa = solve_sa(G, num_reads=NUM_READS)
    out["sa"] = dict(
        cut=sa["cut"],
        ratio=sa["cut"] / classical.cut_value,
        bitstring=sa["bitstring"],
        time_s=time.perf_counter() - t0,
    )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_graph(data: dict) -> None:
    G = nx.Graph()
    G.add_nodes_from(data["graph"]["nodes"])
    G.add_edges_from(data["graph"]["edges"])
    bits = data["bruteforce"]["bitstring"]
    color = ["tab:orange" if bits[i] == "1" else "tab:blue" for i in G.nodes]

    # Mark cut edges vs same-side edges
    edge_colors = []
    edge_widths = []
    for u, v in G.edges:
        if bits[u] != bits[v]:
            edge_colors.append("tab:red")
            edge_widths.append(2.0)
        else:
            edge_colors.append("lightgray")
            edge_widths.append(1.0)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=color, node_size=600, ax=ax,
                           edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    n_part = len(data["bruteforce"]["partitions"])
    ax.set_title(f"Part (a): G(8, 0.5) seed={data['seed']}, "
                 f"Max-Cut = {data['bruteforce']['cut']} "
                 f"({n_part} optimal partitions)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_graph.png", dpi=140)
    plt.close(fig)


def fig_landscape(data: dict) -> None:
    land = data["landscape"]
    gammas = np.array(land["gammas"])
    betas = np.array(land["betas"])
    Z = np.array(land["energy"])  # shape (n_gamma, n_beta)

    fig, ax = plt.subplots(figsize=(7, 5))
    # imshow expects (rows=y, cols=x); transpose so β on y, γ on x
    im = ax.imshow(
        Z.T, origin="lower", aspect="auto",
        extent=[gammas[0], gammas[-1], betas[0], betas[-1]],
        cmap="viridis",
    )
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(rf"Part (b): $\langle H_C \rangle$ landscape at p=1"
                 rf" (min = {land['energy_min']:.3f}"
                 rf" at γ={land['gamma_min']:.2f}, β={land['beta_min']:.2f})")
    ax.scatter([land["gamma_min"]], [land["beta_min"]],
               marker="*", s=250, c="red", edgecolor="white", lw=1.2,
               label="argmin")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label=r"$\langle H_C \rangle$")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_landscape.png", dpi=140)
    plt.close(fig)


def fig_convergence(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(data["qaoa"])))
    for row, c in zip(data["qaoa"], cmap):
        ax.plot(row["energies"], color=c, lw=1.6,
                label=f"p={row['p']}  cut={row['cut']}, r={row['ratio']:.3f}")
    ax.set_xlabel("Adam step")
    ax.set_ylabel(r"$\langle H_C \rangle$  (constant dropped)")
    ax.set_title("Part (c): QAOA convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_qaoa_convergence.png", dpi=140)
    plt.close(fig)


def fig_comparison(data: dict) -> None:
    bf = data["bruteforce"]
    sa = data["sa"]
    qaoa = data["qaoa"]

    methods = ["Brute force", "SA"] + [f"QAOA p={r['p']}" for r in qaoa]
    ratios = [1.0, sa["ratio"]] + [r["ratio"] for r in qaoa]
    cuts = [bf["cut"], sa["cut"]] + [r["cut"] for r in qaoa]
    times = [bf["time_s"], sa["time_s"]] + [r["time_s"] for r in qaoa]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(methods))
    bars = ax1.bar(x, ratios, color="tab:blue", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("Approximation ratio")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(1.0, ls="--", c="k", lw=1, alpha=0.5)
    ax1.set_title(f"Part (d): Method comparison (Max-Cut = {bf['cut']})")
    for b, r, c in zip(bars, ratios, cuts):
        ax1.text(b.get_x() + b.get_width() / 2, r + 0.02,
                 f"{r:.3f}\n({c}/{bf['cut']})",
                 ha="center", fontsize=8)

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
    data = run_all()
    (FIG_DIR / "results.json").write_text(json.dumps(data, indent=2, default=float))

    fig_graph(data)
    fig_landscape(data)
    fig_convergence(data)
    fig_comparison(data)
    print(f"wrote 4 figures + results.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
