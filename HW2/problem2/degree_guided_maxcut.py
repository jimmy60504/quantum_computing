#!/usr/bin/env python3
"""
Degree-Guided Max-Cut Local Search
Phase 1: Fix highest-degree node → put all its neighbors opposite → rest same side
Phase 2: Local search by ascending degree until no improvement
Compare against brute-force optimal across multiple seeds.
"""

import random
from itertools import combinations


# ---------------------------------------------------------------------------
# Graph generation (Erdős–Rényi, pure stdlib)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cut value
# ---------------------------------------------------------------------------

def cut_value(edges: list[tuple], x: list[int]) -> int:
    return sum(1 for u, v in edges if x[u] != x[v])


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------

def brute_force(n: int, edges: list[tuple]) -> tuple[int, list[int]]:
    best, best_x = -1, None
    for mask in range(1 << (n - 1)):          # fix node 0 = 0 (symmetry)
        x = [0] + [(mask >> i) & 1 for i in range(n - 1)]
        c = cut_value(edges, x)
        if c > best:
            best, best_x = c, x[:]
    return best, best_x


# ---------------------------------------------------------------------------
# Degree-Guided Local Search
# ---------------------------------------------------------------------------

def degree_guided_maxcut(n, edges, max_k: int = 5):
    from itertools import combinations as comb

    adj    = adjacency(n, edges)
    adj_set = [set(a) for a in adj]
    degree = [len(adj[v]) for v in range(n)]
    counter = [0]

    # Phase 1 — greedy init
    v_star = max(range(n), key=lambda v: degree[v])
    x = [-1] * n
    x[v_star] = 0
    for u in adj[v_star]:
        x[u] = 1
    for v in range(n):
        if x[v] == -1:
            x[v] = 0
    init_cut = cut_value(edges, x)

    def compute_deltas():
        """ΔC(v) for every node — O(E) total."""
        d = []
        for v in range(n):
            same = sum(1 for u in adj[v] if x[u] == x[v])
            counter[0] += len(adj[v])
            d.append(same - (len(adj[v]) - same))
        return d

    def single_pass(delta):
        flips = 0
        for v in sorted(range(n), key=lambda v: degree[v]):
            if delta[v] > 0:
                x[v] ^= 1
                flips += 1
                # update neighbours' deltas
                for u in adj[v]:
                    delta[u] += 2 if x[u] == x[v] else -2
                delta[v] = -delta[v]
        return flips

    # Phase 2 — single-flip until stuck (with incremental delta)
    delta = compute_deltas()
    while single_pass(delta):
        pass
    max_k_used = 1
    checks_at_k = {1: counter[0]}

    # Phase 3 — k=2: only check cut edges (proven: same-side pairs can't help)
    # ΔC(a,b) = delta[a] + delta[b] + 2  when x[a]≠x[b] and (a,b)∈E
    def double_pass():
        for a, b in edges:
            if x[a] != x[b]:                          # cut edge only
                counter[0] += 1
                if delta[a] + delta[b] + 2 > 0:
                    x[a] ^= 1; x[b] ^= 1
                    # skip mutual edge (a,b): it stays a cut edge, so
                    # contributions to delta[a]/delta[b] are unchanged.
                    # correct formula: -d - 2  (not just -d)
                    for v in (a, b):
                        for u in adj[v]:
                            if u != a and u != b:
                                delta[u] += 2 if x[u] == x[v] else -2
                    delta[a] = -delta[a] - 2
                    delta[b] = -delta[b] - 2
                    return True
        return False

    if cut_value(edges, x) < cut_value(edges, [1-xi for xi in x]):
        pass  # delta still valid

    found2 = True
    while found2:
        found2 = double_pass()
        if found2:
            max_k_used = max(max_k_used, 2)
            while single_pass(delta):   # consolidate after double flip
                pass

    checks_at_k[2] = counter[0]

    # Phase 4 — dense-subgraph guided flip (replaces brute-force k≥3)
    # Seed: each edge. Grow greedily by most-connected node.
    # ΔC(S) only depends on boundary edges → dense S is cheap to evaluate.
    def dense_pass(max_size=8):
        for a, b in edges:
            S = {a, b}
            # grow up to max_size
            for _ in range(max_size - 2):
                scores = {}
                for v in S:
                    for u in adj[v]:
                        if u not in S:
                            scores[u] = scores.get(u, 0) + 1
                if not scores:
                    break
                S.add(max(scores, key=scores.get))
                counter[0] += 1
                # evaluate ΔC(S)
                d = sum(
                    (1 if x[u] == x[v] else -1)
                    for v in S for u in adj[v] if u not in S
                )
                if d > 0:
                    for v in S:
                        x[v] ^= 1
                    return True, len(S)
        return False, 0

    found_dense = True
    while found_dense:
        found_dense, k_used = dense_pass()
        if found_dense:
            max_k_used = max(max_k_used, k_used)
            delta = compute_deltas()
            while single_pass(delta):
                pass
            found2 = True
            while found2:
                found2 = double_pass()
                if found2:
                    max_k_used = max(max_k_used, 2)
                    while single_pass(delta):
                        pass

    checks_at_k['dense'] = counter[0]

    final_cut = cut_value(edges, x)
    info = dict(v_star=v_star, init_cut=init_cut,
                max_k=max_k_used, edge_checks=counter[0],
                checks_at_k=checks_at_k)
    return final_cut, x, info


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

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
        alg, x, info = degree_guided_maxcut(n, edges)
        ratio  = alg / opt if opt > 0 else 1.0
        status = "✓" if alg == opt else f"✗ gap={opt-alg}"
        if alg == opt: wins += 1
        else:          losses += 1

        chk    = info['edge_checks']
        bf_chk = 2**(n-1) * E
        total_checks.append(chk)
        bf_checks.append(bf_chk)

        print(f"{seed:>10} {opt:>5} {alg:>5} {ratio:>6.3f} "
              f"{info['max_k']:>6} {chk:>9,} {bf_chk/chk:>7.1f}x  {status}")

    total = wins + losses
    avg_c  = sum(total_checks) / len(total_checks)
    avg_bf = sum(bf_checks)    / len(bf_checks)
    print(f"\nn={n}: {wins}/{total} optimal ({wins/total:.0%})  "
          f"avg {avg_c:,.0f} vs brute {avg_bf:,.0f}  → {avg_bf/avg_c:.0f}x faster")


if __name__ == "__main__":
    print("=== N=8, p=0.5 ===")
    run(n=8, p=0.5)

    print("\n=== N=12, p=0.5 ===")
    run(n=12, p=0.5)

    print("\n=== N=16, p=0.5 ===")
    run(n=16, p=0.5)
