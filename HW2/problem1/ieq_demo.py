#!/usr/bin/env python3
"""
Iterative Exclusion QUBO (IEQ) — knapsack demonstration.

Starts with 18-qubit slack QUBO. Each round, runs a few-shot SA,
finds the bit with the highest P(x=0), fixes it, and reduces the
problem by one variable. Compares to flat SA baseline with the same
total shot budget.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

WEIGHTS  = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
VALUES   = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72])
CAPACITY = 165
N_ITEMS, N_SLACK = 10, 8
LAM = 8.0
OPT_VALUE = 309


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------

def build_slack_qubo() -> np.ndarray:
    n = N_ITEMS + N_SLACK
    c = np.zeros(n)
    c[:N_ITEMS] = WEIGHTS.astype(float)
    c[N_ITEMS:]  = [2**k for k in range(N_SLACK)]
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = LAM * (c[i]**2 - 2 * CAPACITY * c[i])
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = LAM * 2 * c[i] * c[j]
    for i in range(N_ITEMS):
        Q[i, i] -= VALUES[i]
    return Q                          # upper-triangular QUBO


# ---------------------------------------------------------------------------
# SA engine
# ---------------------------------------------------------------------------

def run_sa(Q_sub: np.ndarray, n_reads: int = 30, n_steps: int = 800,
           T0: float = 5.0, T1: float = 0.01, seed: int = 0) -> np.ndarray:
    """Return (n_reads, n_sub) binary solutions via simulated annealing."""
    rng = np.random.default_rng(seed)
    n = Q_sub.shape[0]
    out = np.zeros((n_reads, n), dtype=np.int8)
    for r in range(n_reads):
        x = rng.integers(0, 2, size=n)
        e = float(x @ Q_sub @ x)
        bx, be = x.copy(), e
        for step in range(n_steps):
            T = T0 * (T1 / T0) ** (step / n_steps)
            i = rng.integers(0, n)
            x[i] ^= 1
            en = float(x @ Q_sub @ x)
            if en < e or rng.random() < np.exp(-(en - e) / max(T, 1e-10)):
                e = en
                if e < be:
                    be, bx = e, x.copy()
            else:
                x[i] ^= 1
        out[r] = bx
    return out


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def feasible_value(x_full: np.ndarray) -> int:
    """Return item value if feasible, else -1."""
    items = x_full[:N_ITEMS]
    if int(WEIGHTS @ items) <= CAPACITY:
        return int(VALUES @ items)
    return -1


# ---------------------------------------------------------------------------
# IEQ: iterative exclusion
# ---------------------------------------------------------------------------

def run_ieq(Q: np.ndarray, reads_per_round: int = 30, rounds: int = 15,
            seed: int = 42) -> list[dict]:
    """
    Iteratively fix the most-likely-zero bit each round.
    Returns per-round history dicts.
    """
    n_full = Q.shape[0]
    active = list(range(n_full))
    history = []

    for rnd in range(rounds):
        if not active:
            break
        idx = np.array(active)
        Q_sub = Q[np.ix_(idx, idx)]

        x_sub_all = run_sa(Q_sub, n_reads=reads_per_round,
                           seed=seed + rnd * 37)

        # Reconstruct full solutions and measure hit rate
        hits, best_val = 0, 0
        for x_sub in x_sub_all:
            x_full = np.zeros(n_full, dtype=np.int8)
            x_full[idx] = x_sub
            v = feasible_value(x_full)
            if v == OPT_VALUE:
                hits += 1
            if v > best_val:
                best_val = v

        # P(x_i = 0) for each active variable
        p_zero = 1.0 - x_sub_all.mean(axis=0)
        fix_local  = int(np.argmax(p_zero))
        fix_global = idx[fix_local]
        bit_type   = "slack" if fix_global >= N_ITEMS else "item"

        row = dict(
            round=rnd + 1,
            n_active=len(active),
            hit_rate=hits / reads_per_round,
            best_value=best_val,
            fix_global=int(fix_global),
            fix_type=bit_type,
            p_zero=float(p_zero[fix_local]),
        )
        history.append(row)
        active.remove(int(fix_global))

        print(f"  Round {rnd+1:2d}: {len(active)+1:2d}→{len(active):2d} bits | "
              f"hit={row['hit_rate']:.0%} val={best_val} | "
              f"fix {bit_type}[{fix_global}] P0={row['p_zero']:.2f}")

    return history


# ---------------------------------------------------------------------------
# Multi-trial hit-rate vs problem size
# ---------------------------------------------------------------------------

def hit_rate_at_size(Q: np.ndarray, fixed_bits: list[int],
                     reads: int = 30, n_trials: int = 40,
                     seed_base: int = 100) -> float:
    n_full = Q.shape[0]
    active = np.array([i for i in range(n_full) if i not in fixed_bits])
    Q_sub = Q[np.ix_(active, active)]
    total_hits = 0
    for t in range(n_trials):
        x_sub_all = run_sa(Q_sub, n_reads=reads,
                           seed=seed_base + t * 41)
        for x_sub in x_sub_all:
            x_full = np.zeros(n_full, dtype=np.int8)
            x_full[active] = x_sub
            if feasible_value(x_full) == OPT_VALUE:
                total_hits += 1
    return total_hits / (n_trials * reads)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("HW2/problem1/figures")
    Q = build_slack_qubo()
    n_full = Q.shape[0]
    READS  = 30
    ROUNDS = 15
    N_TRIALS = 40

    print(f"=== IEQ on {n_full}-qubit slack QUBO (λ={LAM}) ===")
    history = run_ieq(Q, reads_per_round=READS, rounds=ROUNDS, seed=42)
    total_shots = READS * len(history)

    # Multi-trial statistics at selected sizes
    fixed_so_far: list[int] = []
    sizes_to_test = [18, 16, 14, 12, 10]
    size_hr: dict[int, float] = {}

    print(f"\n=== Multi-trial hit rate ({N_TRIALS} trials × {READS} reads) ===")
    for h in history:
        cur_size = h["n_active"]          # size BEFORE fixing this round
        if cur_size in sizes_to_test:
            hr = hit_rate_at_size(Q, fixed_so_far,
                                  reads=READS, n_trials=N_TRIALS)
            size_hr[cur_size] = hr
            print(f"  n_active={cur_size:2d}: hit_rate={hr:.1%}")
        fixed_so_far.append(h["fix_global"])

    # Flat baseline — same total shots, full 18-qubit
    print(f"\n=== Flat baseline ({total_shots} shots, 18 qubits) ===")
    baseline_hits = 0
    for t in range(N_TRIALS):
        x_all = run_sa(Q, n_reads=total_shots, seed=200 + t * 53)
        baseline_hits += sum(feasible_value(x) == OPT_VALUE for x in x_all)
    baseline_hr = baseline_hits / (N_TRIALS * total_shots)
    print(f"  hit_rate={baseline_hr:.1%}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    rounds_x  = [h["round"]    for h in history]
    n_actives = [h["n_active"] for h in history]
    hit_rates = [h["hit_rate"] for h in history]
    bar_colors = ["#e15759" if h["fix_type"] == "slack" else "#4e79a7"
                  for h in history]

    # Panel 1: variables remaining
    ax = axes[0]
    ax.bar(rounds_x, n_actives, color=bar_colors, alpha=0.85,
           edgecolor="white", width=0.7)
    ax.axhline(N_ITEMS, color="gray", ls="--", lw=1)
    ax.text(1.2, N_ITEMS + 0.25, f"N_items = {N_ITEMS}",
            color="gray", fontsize=9)
    ax.set_xlabel("Round")
    ax.set_ylabel("Active variables")
    ax.set_title("Variable reduction path")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e15759", label="slack bit fixed"),
        Patch(color="#4e79a7", label="item bit fixed"),
    ], fontsize=9)

    # Panel 2: hit rate per round
    ax = axes[1]
    ax.bar(rounds_x, [h * 100 for h in hit_rates],
           color="#59a14f", alpha=0.85, edgecolor="white", width=0.7)
    ax.axhline(baseline_hr * 100, color="red", ls="--", lw=1.5,
               label=f"Flat baseline ({baseline_hr:.1%})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Hit rate per round ({READS} shots each)")
    ax.legend(fontsize=9)

    # Panel 3: multi-trial hit rate vs n_active
    ax = axes[2]
    if size_hr:
        xs = sorted(size_hr.keys(), reverse=True)
        ys = [size_hr[s] * 100 for s in xs]
        ax.plot(xs, ys, "o-", color="#4e79a7", lw=2, ms=8)
    ax.axhline(baseline_hr * 100, color="red", ls="--", lw=1.5,
               label=f"Flat 18-qubit ({baseline_hr:.1%})")
    ax.set_xlabel("Active variables (right → smaller problem)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Hit rate as problem shrinks\n({N_TRIALS} trials × {READS} shots)")
    ax.invert_xaxis()
    ax.legend(fontsize=9)

    plt.suptitle("Iterative Exclusion QUBO (IEQ): 18-qubit slack → smaller",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "ieq_reduction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


def build_unbalanced_qubo(l1: float = 0.02, l2: float = 1.61) -> np.ndarray:
    n = N_ITEMS
    w = WEIGHTS.astype(float)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] += l1 * (w[i]**2 - 2 * CAPACITY * w[i])
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += l1 * 2 * w[i] * w[j]
    for i in range(n):
        Q[i, i] += l2 * w[i]
    for i in range(n):
        Q[i, i] -= VALUES[i]
    return Q


def check_value_items(x: np.ndarray) -> int:
    """x is N_ITEMS-length; return feasible value or -1."""
    w = int(WEIGHTS @ x)
    v = int(VALUES @ x)
    return v if w <= CAPACITY else -1


def run_ieq_unbalanced(Q: np.ndarray, reads_per_round: int = 30,
                       rounds: int = 9, seed: int = 42,
                       value_threshold: float = 0.85) -> list[dict]:
    n = Q.shape[0]
    active = list(range(n))
    history = []

    for rnd in range(rounds):
        if not active:
            break
        idx = np.array(active)
        Q_sub = Q[np.ix_(idx, idx)]
        x_sub_all = run_sa(Q_sub, n_reads=reads_per_round,
                           seed=seed + rnd * 37)

        hits, best_val = 0, 0
        for x_sub in x_sub_all:
            x_full = np.zeros(n, dtype=np.int8)
            x_full[idx] = x_sub
            v = check_value_items(x_full)
            if v == OPT_VALUE:
                hits += 1
            if v > best_val:
                best_val = v

        # Only vote from reads whose feasible value >= threshold
        thresh_val = int(value_threshold * OPT_VALUE)
        good_mask = np.array([
            check_value_items(
                np.where(np.isin(np.arange(Q.shape[0]), idx),
                         np.zeros(Q.shape[0], dtype=np.int8),
                         0)
            ) >= thresh_val
            for _ in [None]   # placeholder; compute below
        ])
        # rebuild properly
        good_rows = []
        for x_sub in x_sub_all:
            x_full = np.zeros(Q.shape[0], dtype=np.int8)
            x_full[idx] = x_sub
            v = check_value_items(x_full)
            if v >= thresh_val:
                good_rows.append(x_sub)

        if len(good_rows) >= 3:
            vote_pool = np.array(good_rows)
            label = f"filtered (≥{thresh_val}, n={len(good_rows)})"
        else:
            vote_pool = x_sub_all
            label = f"all reads (too few good: {len(good_rows)})"

        p_zero = 1.0 - vote_pool.mean(axis=0)
        fix_local  = int(np.argmax(p_zero))
        fix_global = idx[fix_local]

        # Stop if no bit is confidently zero
        if p_zero[fix_local] < 0.5:
            print(f"  Round {rnd+1:2d}: stopping — max P0={p_zero[fix_local]:.2f} < 0.5, "
                  f"{len(active)} bits remain → classical solve")
            break

        # Is this the correct decision?
        OPTIMAL_ITEMS = {0, 1, 2, 3, 5}
        correct = fix_global not in OPTIMAL_ITEMS

        row = dict(
            round=rnd + 1,
            n_active=len(active),
            hit_rate=hits / reads_per_round,
            best_value=best_val,
            fix_global=int(fix_global),
            p_zero=float(p_zero[fix_local]),
            correct=correct,
        )
        history.append(row)
        active.remove(int(fix_global))

        marker = "✓" if correct else "✗ WRONG"
        print(f"  Round {rnd+1:2d}: {len(active)+1}→{len(active)} bits | "
              f"hit={row['hit_rate']:.0%} val={best_val} | "
              f"fix item[{fix_global}] P0={row['p_zero']:.2f}  {marker}"
              f"  [{label}]")

    return history


def main_unbalanced():
    out_dir = Path("HW2/problem1/figures")
    Q = build_unbalanced_qubo()
    READS = 30
    ROUNDS = 9   # 10 → 1
    N_TRIALS = 50

    print(f"\n=== IEQ on {N_ITEMS}-qubit UNBALANCED QUBO (λ1=0.02, λ2=1.61) ===")
    history = run_ieq_unbalanced(Q, reads_per_round=READS, rounds=ROUNDS)
    total_shots = READS * len(history)

    # Multi-trial hit rate at each size
    fixed_so_far: list[int] = []
    size_hr: dict[int, float] = {}
    print(f"\n=== Multi-trial hit rate ({N_TRIALS} trials × {READS} reads) ===")
    for h in history:
        cur = h["n_active"]
        n_full = N_ITEMS
        active_idx = np.array([i for i in range(n_full) if i not in fixed_so_far])
        Q_sub = Q[np.ix_(active_idx, active_idx)]
        hits = 0
        for t in range(N_TRIALS):
            xs = run_sa(Q_sub, n_reads=READS, seed=t * 41 + cur)
            for x_sub in xs:
                x_full = np.zeros(n_full, dtype=np.int8)
                x_full[active_idx] = x_sub
                if check_value_items(x_full) == OPT_VALUE:
                    hits += 1
        hr = hits / (N_TRIALS * READS)
        size_hr[cur] = hr
        print(f"  n_active={cur:2d}: hit_rate={hr:.1%}")
        fixed_so_far.append(h["fix_global"])

    # Baseline: flat SA on full 10-qubit unbalanced
    print(f"\n=== Flat baseline ({total_shots} shots, 10 qubits unbalanced) ===")
    b_hits = 0
    for t in range(N_TRIALS):
        xs = run_sa(Q, n_reads=total_shots, seed=300 + t * 53)
        b_hits += sum(check_value_items(x) == OPT_VALUE for x in xs)
    baseline_hr = b_hits / (N_TRIALS * total_shots)
    print(f"  hit_rate={baseline_hr:.1%}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    rounds_x  = [h["round"]    for h in history]
    n_actives = [h["n_active"] for h in history]
    hit_rates = [h["hit_rate"] for h in history]
    bar_colors = ["#59a14f" if h["correct"] else "#e15759" for h in history]

    ax = axes[0]
    ax.bar(rounds_x, n_actives, color=bar_colors, alpha=0.85,
           edgecolor="white", width=0.7)
    ax.set_xlabel("Round")
    ax.set_ylabel("Active variables")
    ax.set_title("Unbalanced QUBO: variable reduction")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#59a14f", label="correct exclusion"),
        Patch(color="#e15759", label="wrong exclusion"),
    ], fontsize=9)

    ax = axes[1]
    ax.bar(rounds_x, [h * 100 for h in hit_rates],
           color="#4e79a7", alpha=0.85, edgecolor="white", width=0.7)
    ax.axhline(baseline_hr * 100, color="red", ls="--", lw=1.5,
               label=f"Flat baseline ({baseline_hr:.1%})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Hit rate per round ({READS} shots each)")
    ax.legend(fontsize=9)

    ax = axes[2]
    if size_hr:
        xs_plot = sorted(size_hr.keys(), reverse=True)
        ys_plot = [size_hr[s] * 100 for s in xs_plot]
        ax.plot(xs_plot, ys_plot, "o-", color="#4e79a7", lw=2, ms=8,
                label="IEQ unbalanced")
    ax.axhline(baseline_hr * 100, color="red", ls="--", lw=1.5,
               label=f"Flat 10-qubit ({baseline_hr:.1%})")
    ax.set_xlabel("Active variables →  smaller")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Hit rate as problem shrinks\n({N_TRIALS} trials × {READS} shots)")
    ax.invert_xaxis()
    ax.legend(fontsize=9)

    plt.suptitle("IEQ on Unbalanced QUBO (10 qubits → smaller)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "ieq_unbalanced.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


def run_one_ieq_trial(Q: np.ndarray, reads_per_round: int,
                      value_threshold: float, seed: int) -> dict:
    """Run one complete IEQ trial. Returns shots_used and whether optimal was found."""
    n = Q.shape[0]
    active = list(range(n))
    total_shots = 0
    OPTIMAL_ITEMS = {0, 1, 2, 3, 5}

    for rnd in range(n - 1):
        if not active:
            break
        idx = np.array(active)
        Q_sub = Q[np.ix_(idx, idx)]
        x_sub_all = run_sa(Q_sub, n_reads=reads_per_round,
                           seed=seed + rnd * 37)
        total_shots += reads_per_round

        # Filter to good reads
        thresh_val = int(value_threshold * OPT_VALUE)
        good_rows = []
        for x_sub in x_sub_all:
            x_full = np.zeros(n, dtype=np.int8)
            x_full[idx] = x_sub
            if check_value_items(x_full) >= thresh_val:
                good_rows.append(x_sub)

        vote_pool = np.array(good_rows) if len(good_rows) >= 3 else x_sub_all
        p_zero = 1.0 - vote_pool.mean(axis=0)

        if p_zero.max() < 0.5:
            break  # no confident zero → stop, solve classically

        fix_global = idx[int(np.argmax(p_zero))]
        active.remove(int(fix_global))

    # Classical brute-force on remaining active bits
    remaining = list(active)
    n_rem = len(remaining)
    best_val = 0
    for mask in range(1 << n_rem):
        x_full = np.zeros(n, dtype=np.int8)
        for bit, glob_idx in enumerate(remaining):
            x_full[glob_idx] = (mask >> bit) & 1
        v = check_value_items(x_full)
        if v > best_val:
            best_val = v

    return dict(
        shots=total_shots,
        n_remaining=n_rem,
        found_optimal=(best_val == OPT_VALUE),
    )


def compare_ieq_vs_flat():
    out_dir = Path("HW2/problem1/figures")
    Q = build_unbalanced_qubo()
    N_TRIALS   = 200
    READS      = 30
    THRESHOLD  = 0.85

    print(f"=== IEQ vs Flat SA comparison ({N_TRIALS} trials) ===\n")

    # --- IEQ trials ---
    ieq_shots, ieq_success, ieq_remaining = [], [], []
    for t in range(N_TRIALS):
        r = run_one_ieq_trial(Q, reads_per_round=READS,
                              value_threshold=THRESHOLD, seed=t * 97 + 1)
        ieq_shots.append(r["shots"])
        ieq_success.append(r["found_optimal"])
        ieq_remaining.append(r["n_remaining"])

    ieq_shots    = np.array(ieq_shots)
    ieq_success  = np.array(ieq_success)
    ieq_remaining = np.array(ieq_remaining)

    print(f"IEQ  — shots per trial: {ieq_shots.mean():.0f} ± {ieq_shots.std():.0f}  "
          f"(min {ieq_shots.min()}, max {ieq_shots.max()})")
    print(f"IEQ  — remaining bits when stopped: {ieq_remaining.mean():.1f} ± {ieq_remaining.std():.1f}")
    print(f"IEQ  — overall success rate: {ieq_success.mean():.1%}")
    print(f"IEQ  — avg shots to brute-force boundary: {ieq_shots.mean():.0f} → 2^{ieq_remaining.mean():.1f} classical")

    # --- Flat SA at equivalent budgets ---
    budgets = sorted(set(ieq_shots.tolist()))
    budget_hr = {}
    for B in budgets:
        hits = 0
        for t in range(N_TRIALS):
            xs = run_sa(Q, n_reads=B, seed=t * 53 + 500)
            hits += any(check_value_items(x) == OPT_VALUE for x in xs)
        budget_hr[B] = hits / N_TRIALS

    print(f"\nFlat SA —")
    for B, hr in sorted(budget_hr.items()):
        print(f"  {B:4d} shots: {hr:.1%}")

    # --- Sweep: P(success) vs total shots ---
    shot_range = np.arange(30, 330, 30)
    flat_hr_sweep = []
    for B in shot_range:
        hits = 0
        for t in range(N_TRIALS):
            xs = run_sa(Q, n_reads=int(B), seed=t * 71 + 700)
            hits += any(check_value_items(x) == OPT_VALUE for x in xs)
        flat_hr_sweep.append(hits / N_TRIALS)

    # IEQ P(success | shots ≤ B): cumulative
    ieq_cumhr = [(ieq_shots <= B).mean() * ieq_success[ieq_shots <= B].mean()
                 if (ieq_shots <= B).any() else 0.0
                 for B in shot_range]
    # Simpler: just use overall IEQ success rate at each budget threshold
    ieq_cumhr = [ieq_success[ieq_shots <= B].mean() if (ieq_shots <= B).any() else 0.0
                 for B in shot_range]

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: shots distribution for IEQ
    ax = axes[0]
    vals, cnts = np.unique(ieq_shots, return_counts=True)
    ax.bar(vals, cnts / N_TRIALS * 100, width=20, color="#4e79a7", alpha=0.85)
    ax.set_xlabel("Shots used by IEQ")
    ax.set_ylabel("Fraction of trials (%)")
    ax.set_title("IEQ: shots per trial distribution")

    # Panel 2: remaining bits distribution
    ax = axes[1]
    vals2, cnts2 = np.unique(ieq_remaining, return_counts=True)
    colors2 = ["#59a14f" if v <= 5 else "#e15759" for v in vals2]
    ax.bar(vals2, cnts2 / N_TRIALS * 100, color=colors2, alpha=0.85, width=0.6)
    ax.set_xlabel("Bits remaining at stop")
    ax.set_ylabel("Fraction of trials (%)")
    ax.set_title("IEQ: size of remaining classical subproblem")
    ax.axvline(5, color="gray", ls="--", lw=1, label="optimal size = 5")
    ax.legend(fontsize=9)

    # Panel 3: P(success) vs shots budget
    ax = axes[2]
    ax.plot(shot_range, [h * 100 for h in flat_hr_sweep],
            "o-", color="red", lw=2, ms=6, label="Flat SA")
    ax.plot(shot_range, [h * 100 for h in ieq_cumhr],
            "s-", color="#4e79a7", lw=2, ms=6, label="IEQ + brute-force")
    ax.set_xlabel("Total shots budget")
    ax.set_ylabel("P(find optimal) (%)")
    ax.set_title("Success rate vs shot budget")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)

    plt.suptitle("IEQ (Unbalanced QUBO) vs Flat SA — knapsack N=10",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "ieq_vs_flat.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    compare_ieq_vs_flat()
