#!/usr/bin/env python3
"""
Chain-Crystal Max-Cut
=====================
Physical analogy: a densely-connected subgraph is a "crystal".
At a 1-opt local optimum, a misaligned crystal must be rotated (flipped)
as a whole unit — flipping any subset of it makes things worse.

Key insight: ΔC(S) depends only on the BOUNDARY edges of S.
Internal edges (both endpoints in S) cancel out entirely.
So a dense S has a small, well-defined boundary → easy to evaluate.

Finding the crystal: start from a seed node, greedily absorb the
adjacent node that maximally improves ΔC(S) at each step ("best-delta").
This traces the path of least resistance through the energy barrier —
the same way a crystal rotates by pivoting along its softest bond.

Algorithm
---------
  Phase 1  Greedy init
             v* = highest-degree node, fixed to side 0.
             All neighbours of v* → side 1.
             Remaining nodes → side 0.
             Breaks symmetry and creates a strong initial cut around v*.

  Phase 2  1-opt refinement
             Flip any node v where ΔC(v) = |same-side neighbours| − |opposite| > 0.
             Repeat until no single flip improves the cut.

  Phase 3  Chain-crystal search
             For each seed node (tried in descending degree order):
               Build crystal S = {seed}, track ΔC(S).
               Repeat: find the adjacent node u that maximises
                       δ(u,S) = ΔC(S∪{u}) − ΔC(S)
                              = Σ_{w∈N(u)} sign(x[w]=x[u]) × (−1 if w∈S else +1)
               Add best u to S, update ΔC(S) += δ(u,S).
               If ΔC(S) > 0: flip S, go back to Phase 2, restart Phase 3.
             If no seed yields ΔC > 0 within max_chain steps: done.

Complexity (per outer iteration)
  Phase 2: O(N · E / N) = O(E) per pass, O(E · passes) total
  Phase 3: O(N · max_chain · avg_degree) = O(N · E / N · max_chain) = O(E · max_chain)
"""

import random


# ── Graph utilities ──────────────────────────────────────────────────────────

def gnp_graph(n: int, p: float, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if rng.random() < p]


def adjacency(n: int, edges: list[tuple]) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def cut_value(edges: list[tuple], x: list[int]) -> int:
    return sum(1 for u, v in edges if x[u] != x[v])


# ── Brute force (reference) ──────────────────────────────────────────────────

def brute_force(n: int, edges: list[tuple]) -> tuple[int, list[int]]:
    best, best_x = -1, None
    for mask in range(1 << (n - 1)):          # fix node 0 = side 0 (symmetry)
        x = [0] + [(mask >> i) & 1 for i in range(n - 1)]
        c = cut_value(edges, x)
        if c > best:
            best, best_x = c, x[:]
    return best, best_x


# ── Chain-Crystal Max-Cut ────────────────────────────────────────────────────

def chain_crystal_maxcut(
    n: int,
    edges: list[tuple],
    max_chain: int = 12,
) -> tuple[int, list[int], dict]:

    adj    = adjacency(n, edges)
    degree = [len(adj[v]) for v in range(n)]
    checks = [0]

    # ── Phase 1: greedy init ─────────────────────────────────────────────────
    v_star = max(range(n), key=lambda v: degree[v])
    x = [0] * n
    for u in adj[v_star]:
        x[u] = 1

    # ── Phase 2: 1-opt ───────────────────────────────────────────────────────
    def one_opt():
        improved = True
        while improved:
            improved = False
            # low-degree first: cheaper to evaluate, fewer cascade effects
            for v in sorted(range(n), key=lambda v: degree[v]):
                same = sum(1 for u in adj[v] if x[u] == x[v])
                checks[0] += degree[v]
                if same - (degree[v] - same) > 0:
                    x[v] ^= 1
                    improved = True

    # ── Phase 3: chain-crystal search ───────────────────────────────────────
    def delta_inc(u: int, S_set: set) -> int:
        """ΔC change when absorbing node u into crystal S."""
        d = 0
        for w in adj[u]:
            sign = 1 if x[w] == x[u] else -1
            d += sign if w not in S_set else -sign
        checks[0] += degree[u]
        return d

    def chain_pass() -> tuple[bool, int]:
        """
        Try each node as a crystal seed.
        Grow by absorbing the neighbour with the best delta_inc.
        Return (improved, crystal_size) — True if a flip was applied.
        """
        for seed_v in sorted(range(n), key=lambda v: -degree[v]):
            S     = {seed_v}
            dc    = sum((1 if x[u] == x[v] else -1)
                        for v in S for u in adj[v] if u not in S)
            checks[0] += degree[seed_v]

            for _ in range(max_chain - 1):
                # build frontier: nodes adjacent to S
                frontier = {}
                for v in S:
                    for u in adj[v]:
                        if u not in S:
                            frontier[u] = frontier.get(u, 0) + 1

                if not frontier:
                    break

                # precompute delta_inc for all candidates; pick best (no double call)
                scores = {u: delta_inc(u, S) for u in frontier}
                best   = max(scores, key=scores.__getitem__)
                dc    += scores[best]
                S.add(best)

                if dc > 0:
                    for v in S:
                        x[v] ^= 1
                    return True, len(S)

        return False, 0

    # ── Main loop ────────────────────────────────────────────────────────────
    one_opt()
    max_k_used = 1

    while True:
        improved, k = chain_pass()
        if not improved:
            break
        max_k_used = max(max_k_used, k)
        one_opt()

    return (
        cut_value(edges, x),
        x,
        dict(max_k=max_k_used, checks=checks[0]),
    )


# ── Benchmark ────────────────────────────────────────────────────────────────

def run(n: int = 8, p: float = 0.5, seeds: list[int] = None):
    if seeds is None:
        seeds = [11224001, 42, 7, 2025, 999, 12345, 314159, 271828, 100, 55]

    header = (f"{'seed':>10} {'opt':>5} {'alg':>5} {'ratio':>6} "
              f"{'max_k':>6} {'checks':>9} {'vs BF':>8}  status")
    print(header)
    print("─" * len(header))

    wins = losses = 0
    total_checks, bf_checks = [], []

    for seed in seeds:
        edges = gnp_graph(n, p, seed)
        if not edges:
            continue
        E = len(edges)

        opt, _       = brute_force(n, edges)
        alg, x, info = chain_crystal_maxcut(n, edges)
        ratio  = alg / opt if opt > 0 else 1.0
        status = "✓" if alg == opt else f"✗ gap={opt - alg}"
        if alg == opt:
            wins += 1
        else:
            losses += 1

        chk    = info['checks']
        bf_chk = 2 ** (n - 1) * E
        total_checks.append(chk)
        bf_checks.append(bf_chk)

        print(f"{seed:>10} {opt:>5} {alg:>5} {ratio:>6.3f} "
              f"{info['max_k']:>6} {chk:>9,} {bf_chk / chk:>7.1f}x  {status}")

    total  = wins + losses
    avg_c  = sum(total_checks) / len(total_checks)
    avg_bf = sum(bf_checks)    / len(bf_checks)
    print(f"\nn={n}: {wins}/{total} optimal ({wins / total:.0%})  "
          f"avg {avg_c:,.0f} vs brute {avg_bf:,.0f}  → {avg_bf / avg_c:.0f}x faster")


if __name__ == "__main__":
    for n in [8, 12, 16]:
        print(f"\n=== N={n}, p=0.5 ===")
        run(n=n)
