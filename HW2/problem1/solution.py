"""HW2 Problem 1 — 01 Knapsack: QUBO and Quantum Annealing."""

from __future__ import annotations

import argparse
import itertools
import time
from typing import NamedTuple

import dimod
import neal
import numpy as np

# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------

SEED = 0  # TODO: replace with your student ID

WEIGHTS = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
VALUES  = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
N_ITEMS = len(WEIGHTS)
CAPACITY = 165

# Slack variables: M = ceil(log2(W)) = 8  →  18 total binary variables
N_SLACK = int(np.ceil(np.log2(CAPACITY)))  # 8
N_VARS  = N_ITEMS + N_SLACK               # 18

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Part (a) — Classical solver
# ---------------------------------------------------------------------------

class ClassicalSolution(NamedTuple):
    selected: list[int]   # 0-based item indices
    total_weight: int
    total_value: int


def solve_brute_force() -> ClassicalSolution:
    """Enumerate all 2^N subsets and return the feasible one with max value."""
    best_value  = -1
    best_mask   = 0
    for mask in range(1 << N_ITEMS):
        bits    = np.array([(mask >> i) & 1 for i in range(N_ITEMS)])
        weight  = int(WEIGHTS @ bits)
        value   = int(VALUES  @ bits)
        if weight <= CAPACITY and value > best_value:
            best_value = value
            best_mask  = mask
    bits     = [(best_mask >> i) & 1 for i in range(N_ITEMS)]
    selected = [i for i, b in enumerate(bits) if b]
    return ClassicalSolution(
        selected     = selected,
        total_weight = int(WEIGHTS[selected].sum()),
        total_value  = int(VALUES[selected].sum()),
    )


# ---------------------------------------------------------------------------
# Part (a) — QUBO construction
# ---------------------------------------------------------------------------

def build_qubo(lam: float) -> dict[tuple[int, int], float]:
    """Return the QUBO dict Q for the given penalty coefficient lambda.

    Variable layout
    ---------------
    x_0 … x_{N_ITEMS-1}  : item selection bits
    s_0 … s_{N_SLACK-1}  : slack bits  (s_k represents 2^k)

    QUBO objective (slack-variable method):
        min_{x,s}  -∑ v_i x_i  +  λ (∑ w_i x_i + ∑ 2^k s_k − W)²
    """
    Q: dict[tuple[int, int], float] = {}

    def add(i: int, j: int, val: float) -> None:
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + val

    # Coefficient vector: c_i = w_i for items, c_{N_ITEMS+k} = 2^k for slack
    c = np.empty(N_VARS)
    c[:N_ITEMS]  = WEIGHTS
    c[N_ITEMS:]  = 2 ** np.arange(N_SLACK)

    # Expand  λ (∑ c_i q_i − W)²
    # = λ [ ∑_i (c_i² - 2W c_i) q_i  +  2 ∑_{i<j} c_i c_j q_i q_j  +  W² ]
    for i in range(N_VARS):
        add(i, i, lam * (c[i] ** 2 - 2 * CAPACITY * c[i]))
    for i in range(N_VARS):
        for j in range(i + 1, N_VARS):
            add(i, j, lam * 2 * c[i] * c[j])

    # Objective: -∑ v_i x_i  (diagonal terms for item bits only)
    for i in range(N_ITEMS):
        add(i, i, -float(VALUES[i]))

    return Q


# ---------------------------------------------------------------------------
# Part (b) — ExactSolver + λ sweep
# ---------------------------------------------------------------------------

def solve_exact(
    Q: dict[tuple[int, int], float],
) -> tuple[dict[int, int], float]:
    """Run ExactSolver and return (best sample, energy)."""
    bqm    = dimod.BQM.from_qubo(Q)
    result = dimod.ExactSolver().sample(bqm)
    best   = result.first
    return dict(best.sample), best.energy


def sweep_lambda(
    lambdas: list[float],
    classical: ClassicalSolution,
) -> list[dict]:
    """For each λ, build QUBO, run ExactSolver, check feasibility and optimality."""
    rows = []
    for lam in lambdas:
        Q               = build_qubo(lam)
        sample, energy  = solve_exact(Q)
        x_bits          = np.array([sample[i] for i in range(N_ITEMS)])
        weight          = int(WEIGHTS @ x_bits)
        value           = int(VALUES  @ x_bits)
        feasible        = weight <= CAPACITY
        optimal         = feasible and value == classical.total_value
        rows.append(dict(
            lam      = lam,
            weight   = weight,
            value    = value,
            energy   = energy,
            feasible = feasible,
            optimal  = optimal,
        ))
    return rows


# ---------------------------------------------------------------------------
# Part (c) — SimulatedAnnealingSampler + num_reads sweep
# ---------------------------------------------------------------------------

def solve_sa(
    Q: dict[tuple[int, int], float],
    num_reads: int,
) -> tuple[dict[int, int], float]:
    """Run SimulatedAnnealingSampler and return (best sample, energy)."""
    bqm    = dimod.BQM.from_qubo(Q)
    result = neal.SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, seed=SEED
    )
    best   = result.first
    return dict(best.sample), best.energy


def sweep_num_reads(
    num_reads_list: list[int],
    Q: dict[tuple[int, int], float],
    optimal_bits: list[int],
) -> list[dict]:
    """For each num_reads value, run SA and report success probability."""
    rows = []
    bqm  = dimod.BQM.from_qubo(Q)
    for num_reads in num_reads_list:
        t0 = time.perf_counter()
        result = neal.SimulatedAnnealingSampler().sample(
            bqm, num_reads=num_reads, seed=SEED
        )
        elapsed = time.perf_counter() - t0
        samples_x = []
        sample_values = []
        successes = 0
        for s, _, _ in result.data(["sample", "energy", "num_occurrences"]):
            bits = [s[i] for i in range(N_ITEMS)]
            samples_x.append(bits)
            sample_values.append(int(np.array(bits) @ VALUES))
            if bits == optimal_bits:
                successes += 1
        best_x = np.array([result.first.sample[i] for i in range(N_ITEMS)])
        rows.append(dict(
            num_reads    = num_reads,
            success_prob = successes / num_reads,
            best_value   = int(VALUES  @ best_x),
            best_weight  = int(WEIGHTS @ best_x),
            time_s       = elapsed,
            sample_values = sample_values,
        ))
    return rows


# ---------------------------------------------------------------------------
# Part (d) — Comparison table
# ---------------------------------------------------------------------------

def print_comparison(
    classical: ClassicalSolution,
    exact_row: dict,
    sa_rows: list[dict],
    times: dict[str, float],
) -> None:
    header = f"{'Method':<30} {'Value':>6} {'Weight':>7} {'Feasible':>9} {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    def row(name: str, value: int, weight: int, feasible: bool, t: float) -> None:
        print(f"{name:<30} {value:>6} {weight:>7} {str(feasible):>9} {t:>10.4f}")

    row("Classical (brute-force)",
        classical.total_value, classical.total_weight, True, times["classical"])
    row(f"Exact QUBO (λ={exact_row['lam']})",
        exact_row["value"], exact_row["weight"], exact_row["feasible"], times["exact"])
    for r in sa_rows:
        feasible = r["best_weight"] <= CAPACITY
        row(f"SA (num_reads={r['num_reads']})",
            r["best_value"], r["best_weight"], feasible, times.get(f"sa_{r['num_reads']}", 0.0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW2 Problem 1 — 01 Knapsack")
    p.add_argument("--lam", type=float, nargs="+", default=[4.0, 8.0, 16.0],
                   help="Penalty coefficients λ to sweep (part b)")
    p.add_argument("--num-reads", type=int, nargs="+", default=[10, 100, 1000, 10000],
                   help="num_reads values to sweep (part c)")
    p.add_argument("--best-lam", type=float, default=8.0,
                   help="λ used for the SA sweep and comparison table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    times: dict[str, float] = {}

    # --- (a) Classical ---
    t0 = time.perf_counter()
    classical = solve_brute_force()
    times["classical"] = time.perf_counter() - t0
    print(f"\n[Part a] Classical optimum")
    print(f"  Items (1-based): {[i+1 for i in classical.selected]}")
    print(f"  Total weight   : {classical.total_weight}")
    print(f"  Total value    : {classical.total_value}")

    # --- (b) ExactSolver λ sweep ---
    print(f"\n[Part b] ExactSolver λ sweep: {args.lam}")
    t0 = time.perf_counter()
    exact_rows = sweep_lambda(args.lam, classical)
    times["exact"] = time.perf_counter() - t0
    for r in exact_rows:
        status = "optimal" if r["optimal"] else ("feasible" if r["feasible"] else "infeasible")
        print(f"  λ={r['lam']:6.1f}  value={r['value']:4d}  weight={r['weight']:4d}  {status}")

    # Choose the best-λ row for part (d)
    best_exact_row = next(r for r in exact_rows if r["lam"] == args.best_lam)

    # --- (c) SA num_reads sweep ---
    print(f"\n[Part c] SA num_reads sweep: {args.num_reads}")
    Q = build_qubo(args.best_lam)
    # Optimal bitstring from classical solution
    optimal_bits = [1 if i in classical.selected else 0 for i in range(N_ITEMS)]
    sa_rows = sweep_num_reads(args.num_reads, Q, optimal_bits)
    for r in sa_rows:
        times[f"sa_{r['num_reads']}"] = r["time_s"]
    for r in sa_rows:
        print(f"  num_reads={r['num_reads']:6d}  success_prob={r['success_prob']:.4f}  best_value={r['best_value']}")

    # --- (d) Comparison table ---
    print(f"\n[Part d] Comparison table")
    print_comparison(classical, best_exact_row, sa_rows, times)


if __name__ == "__main__":
    main()
