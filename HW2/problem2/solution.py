"""HW2 Problem 2 — Max-Cut on a random 8-node graph via QAOA (PennyLane)."""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------

SEED = 0  # TODO: replace with your student ID
N_NODES = 8
EDGE_PROB = 0.5


def build_graph(seed: int = SEED) -> nx.Graph:
    return nx.gnp_random_graph(n=N_NODES, p=EDGE_PROB, seed=seed)


# ---------------------------------------------------------------------------
# Part (a) — Brute-force Max-Cut
# ---------------------------------------------------------------------------

class CutSolution(NamedTuple):
    bitstring: str            # length-N string of '0'/'1'
    cut_value: int
    partitions: list[str]     # all bitstrings achieving the optimum


def cut_value(G: nx.Graph, bits: np.ndarray) -> int:
    """Number of edges crossing the partition defined by bits ∈ {0,1}^N."""
    return sum(1 for i, j in G.edges if bits[i] != bits[j])


def brute_force_maxcut(G: nx.Graph) -> CutSolution:
    n = G.number_of_nodes()
    best = -1
    best_strings: list[str] = []
    for mask in range(1 << n):
        bits = np.array([(mask >> i) & 1 for i in range(n)])
        v = cut_value(G, bits)
        s = "".join(str(b) for b in bits)
        if v > best:
            best = v
            best_strings = [s]
        elif v == best:
            best_strings.append(s)
    return CutSolution(best_strings[0], best, best_strings)


# ---------------------------------------------------------------------------
# QAOA building blocks
# ---------------------------------------------------------------------------

def cost_hamiltonian(G: nx.Graph) -> qml.Hamiltonian:
    """H_C = +1/2 ∑_{(i,j)∈E} Z_i Z_j  − |E|/2 · I  (constant dropped).

    Ground state = max-cut bitstring, so QAOA can MIN ⟨H_C⟩.
    ⟨H_C⟩ + |E|/2 = -cut_value.
    """
    coeffs = [0.5] * G.number_of_edges()
    ops = [qml.PauliZ(i) @ qml.PauliZ(j) for i, j in G.edges]
    return qml.Hamiltonian(coeffs, ops)


def mixer_hamiltonian(n: int) -> qml.Hamiltonian:
    return qml.Hamiltonian([1.0] * n, [qml.PauliX(i) for i in range(n)])


def make_qaoa_circuit(G: nx.Graph, p: int):
    n = G.number_of_nodes()
    H_C = cost_hamiltonian(G)
    H_M = mixer_hamiltonian(n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="autograd")
    def expval(params):
        gammas, betas = params[0], params[1]
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p):
            qml.exp(H_C, -1j * gammas[layer])
            qml.exp(H_M, -1j * betas[layer])
        return qml.expval(H_C)

    @qml.qnode(dev, interface="autograd")
    def probs(params):
        gammas, betas = params[0], params[1]
        for w in range(n):
            qml.Hadamard(wires=w)
        for layer in range(p):
            qml.exp(H_C, -1j * gammas[layer])
            qml.exp(H_M, -1j * betas[layer])
        return qml.probs(wires=range(n))

    return expval, probs


# ---------------------------------------------------------------------------
# Part (b) — Energy landscape at p=1
# ---------------------------------------------------------------------------

def landscape_p1(G: nx.Graph, n_grid: int = 40) -> dict:
    """Return dict with γ grid, β grid, and F(γ,β) heatmap values."""
    expval, _ = make_qaoa_circuit(G, p=1)
    gammas = np.linspace(0, 2 * np.pi, n_grid)
    betas  = np.linspace(0, np.pi,     n_grid)
    Z = np.empty((n_grid, n_grid))
    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            params = pnp.array([[g], [b]], requires_grad=False)
            Z[i, j] = float(expval(params))
    idx = np.unravel_index(np.argmin(Z), Z.shape)
    return dict(
        gammas=gammas, betas=betas, energy=Z,
        gamma_min=float(gammas[idx[0]]),
        beta_min=float(betas[idx[1]]),
        energy_min=float(Z[idx]),
    )


# ---------------------------------------------------------------------------
# Part (c) — QAOA depth sweep
# ---------------------------------------------------------------------------

def run_qaoa(
    G: nx.Graph,
    p: int,
    n_steps: int = 200,
    stepsize: float = 0.1,
    seed: int = SEED,
) -> dict:
    expval, probs = make_qaoa_circuit(G, p)

    rng = np.random.default_rng(seed)
    init = pnp.array(rng.uniform(0, np.pi, size=(2, p)), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=stepsize)

    params = init
    energies = []
    for _ in range(n_steps):
        params, e = opt.step_and_cost(expval, params)
        energies.append(float(e))

    pr = probs(params)
    best_idx = int(np.argmax(pr))
    n = G.number_of_nodes()
    # PennyLane probs index: wire 0 is MSB
    bits = np.array([(best_idx >> (n - 1 - i)) & 1 for i in range(n)])
    bitstring = "".join(str(b) for b in bits)
    cut = cut_value(G, bits)

    return dict(
        p=p,
        params_opt=np.asarray(params),
        energies=energies,
        bitstring=bitstring,
        cut=cut,
        prob_best=float(pr[best_idx]),
    )


# ---------------------------------------------------------------------------
# Part (d) — SA baseline (dimod / neal)
# ---------------------------------------------------------------------------

def solve_sa(G: nx.Graph, num_reads: int = 1000, seed: int = SEED) -> dict:
    import dimod
    import neal

    # Max-Cut → minimise ∑_{(i,j)∈E} Z_i Z_j  (anti-ferromagnetic, J_ij = +1)
    h = {i: 0.0 for i in G.nodes}
    J = {(i, j): 1.0 for i, j in G.edges}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    res = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, seed=seed)
    best = res.first
    bits = np.array([(1 - best.sample[i]) // 2 for i in G.nodes])  # spin → bit
    return dict(
        bitstring="".join(str(b) for b in bits),
        cut=cut_value(G, bits),
        energy=float(best.energy),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-list", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-grid", type=int, default=40)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--num-reads", type=int, default=1000)
    p.add_argument("--out-dir", default="HW2/problem2/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    G = build_graph()
    times: dict[str, float] = {}

    # (a)
    t0 = time.perf_counter()
    classical = brute_force_maxcut(G)
    times["bruteforce"] = time.perf_counter() - t0
    print(f"\n[Part a] |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
    print(f"  Max-cut value : {classical.cut_value}")
    print(f"  Optimum bitstrings ({len(classical.partitions)}): {classical.partitions[:4]}{'...' if len(classical.partitions) > 4 else ''}")

    # (b)
    print(f"\n[Part b] p=1 landscape on {args.n_grid}×{args.n_grid} grid")
    t0 = time.perf_counter()
    land = landscape_p1(G, n_grid=args.n_grid)
    times["landscape"] = time.perf_counter() - t0
    print(f"  argmin: γ={land['gamma_min']:.3f}, β={land['beta_min']:.3f}, ⟨H_C⟩={land['energy_min']:.4f}")

    # (c)
    print(f"\n[Part c] QAOA depth sweep p={args.p_list}")
    qaoa_rows = []
    for p in args.p_list:
        t0 = time.perf_counter()
        row = run_qaoa(G, p, n_steps=args.n_steps)
        times[f"qaoa_p{p}"] = time.perf_counter() - t0
        qaoa_rows.append(row)
        print(f"  p={p}  cut={row['cut']}/{classical.cut_value}  r={row['cut']/classical.cut_value:.3f}  bits={row['bitstring']}")

    # (d)
    print(f"\n[Part d] SA baseline (num_reads={args.num_reads})")
    t0 = time.perf_counter()
    sa = solve_sa(G, num_reads=args.num_reads)
    times["sa"] = time.perf_counter() - t0
    print(f"  cut={sa['cut']}/{classical.cut_value}  r={sa['cut']/classical.cut_value:.3f}")

    print(f"\n[Part d] Comparison")
    header = f"{'Method':<22} {'Cut':>5} {'Ratio':>7} {'Time (s)':>10}"
    print(header); print("-" * len(header))
    print(f"{'Brute force':<22} {classical.cut_value:>5} {1.000:>7.3f} {times['bruteforce']:>10.4f}")
    print(f"{'SA':<22} {sa['cut']:>5} {sa['cut']/classical.cut_value:>7.3f} {times['sa']:>10.4f}")
    for row in qaoa_rows:
        name = f"QAOA p={row['p']}"
        r = row['cut'] / classical.cut_value
        print(f"{name:<22} {row['cut']:>5} {r:>7.3f} {times[f'qaoa_p{row['p']}']:>10.4f}")


if __name__ == "__main__":
    main()
