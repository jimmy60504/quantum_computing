"""HW2 Problem 3 — LABS for N=20.

Strategies implemented:
    1. Quartic-Hamiltonian QAOA  (PennyLane, depth p)
    2. VQE with hardware-efficient ansatz (RY layers + ring CNOTs)

Baselines:
    - Random sampling under the same shot budget
    - Simulated annealing (neal) on a quadratized QUBO
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Iterable, NamedTuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Problem data / cost function
# ---------------------------------------------------------------------------

SEED = 0  # TODO: replace with your student ID
N_DEFAULT = 20

BARKER_11 = np.array([+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1])  # E=5, F=12.10


def labs_energy(s: np.ndarray) -> int:
    """E(s) = sum_{k=1}^{N-1} C_k(s)^2,  C_k = sum_i s_i s_{i+k}."""
    s = np.asarray(s, dtype=int)
    N = len(s)
    e = 0
    for k in range(1, N):
        ck = int(np.dot(s[:N - k], s[k:]))
        e += ck * ck
    return e


def merit_factor(s: np.ndarray) -> float:
    N = len(s)
    return N * N / (2.0 * labs_energy(s))


def verify_barker11() -> None:
    e = labs_energy(BARKER_11)
    f = merit_factor(BARKER_11)
    assert e == 5, f"expected E=5, got {e}"
    assert abs(f - 12.10) < 0.01, f"expected F≈12.10, got {f:.4f}"
    print(f"[verify] N=11 Barker: E={e}, F={f:.4f}  ✓")


def bits_to_spins(bits: Iterable[int]) -> np.ndarray:
    return np.where(np.asarray(bits) == 0, +1, -1)


# ---------------------------------------------------------------------------
# Quartic LABS Hamiltonian
# ---------------------------------------------------------------------------

def labs_pauli_terms(N: int) -> tuple[list[float], list[tuple[int, ...]]]:
    """Build coefficients/qubit-tuples for H_C^LABS = ∑_k ∑_{i,j} Z_i Z_{i+k} Z_j Z_{j+k}.

    Repeated Z factors cancel; the constant offset is dropped (returned separately).
    Returns (coeffs, supports) for non-identity Pauli-Z products only.
    """
    from collections import Counter

    counter: Counter[tuple[int, ...]] = Counter()
    constant = 0
    for k in range(1, N):
        for i, j in itertools.product(range(N - k), repeat=2):
            qubits = (i, i + k, j, j + k)
            # reduce repeated Z factors mod 2
            c: Counter[int] = Counter(qubits)
            reduced = tuple(sorted(q for q, m in c.items() if m % 2))
            if not reduced:
                constant += 1
            else:
                counter[reduced] += 1
    coeffs = [float(v) for v in counter.values()]
    supports = list(counter.keys())
    return coeffs, supports, constant  # type: ignore[return-value]


def cost_hamiltonian(N: int) -> tuple[qml.Hamiltonian, float]:
    coeffs, supports, const = labs_pauli_terms(N)  # type: ignore[misc]
    ops = []
    for sup in supports:
        op = qml.PauliZ(sup[0])
        for q in sup[1:]:
            op = op @ qml.PauliZ(q)
        ops.append(op)
    return qml.Hamiltonian(coeffs, ops), float(const)


# ---------------------------------------------------------------------------
# Strategy 1 — Quartic QAOA
# ---------------------------------------------------------------------------

def make_qaoa(N: int, p: int):
    H_C, const = cost_hamiltonian(N)
    H_M = qml.Hamiltonian([1.0] * N, [qml.PauliX(i) for i in range(N)])
    dev = qml.device("default.qubit", wires=N)

    def circuit(params):
        g, b = params[0], params[1]
        for w in range(N):
            qml.Hadamard(wires=w)
        for layer in range(p):
            qml.exp(H_C, -1j * g[layer])
            qml.exp(H_M, -1j * b[layer])

    @qml.qnode(dev, interface="autograd")
    def energy(params):
        circuit(params)
        return qml.expval(H_C)

    @qml.qnode(dev)
    def probs(params):
        circuit(params)
        return qml.probs(wires=range(N))

    return energy, probs, const


def _decode_best(probs_vec: np.ndarray, N: int, top_k: int = 64) -> tuple[int, np.ndarray]:
    """Take top-K basis states by probability and return the lowest-E one."""
    K = min(top_k, len(probs_vec))
    top = np.argsort(probs_vec)[-K:]
    best_e, best_s = None, None
    for idx in top:
        bits = np.array([(idx >> (N - 1 - i)) & 1 for i in range(N)])
        s = bits_to_spins(bits)
        e = labs_energy(s)
        if best_e is None or e < best_e:
            best_e, best_s = e, s
    return best_e, best_s


def run_qaoa_labs(
    N: int,
    p: int,
    n_steps: int = 100,
    stepsize: float = 0.05,
    seed: int = SEED,
) -> dict:
    energy, probs, const = make_qaoa(N, p)
    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, np.pi, size=(2, p)), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    trace = []
    for _ in range(n_steps):
        params, e = opt.step_and_cost(energy, params)
        trace.append(float(e) + const)

    pr = np.asarray(probs(params))
    best_e, best_s = _decode_best(pr, N)
    return dict(strategy=f"QAOA-quartic p={p}", best_E=best_e, best_s=best_s,
                trace=trace, n_eval=n_steps)


# ---------------------------------------------------------------------------
# Strategy 2 — VQE with hardware-efficient ansatz
# ---------------------------------------------------------------------------

def make_vqe(N: int, layers: int):
    H_C, const = cost_hamiltonian(N)
    dev = qml.device("default.qubit", wires=N)

    def ansatz(params):
        # params shape: (layers+1, N)
        for w in range(N):
            qml.RY(params[0, w], wires=w)
        for L in range(layers):
            for w in range(N):
                qml.CNOT(wires=[w, (w + 1) % N])
            for w in range(N):
                qml.RY(params[L + 1, w], wires=w)

    @qml.qnode(dev, interface="autograd")
    def energy(params):
        ansatz(params)
        return qml.expval(H_C)

    @qml.qnode(dev)
    def probs(params):
        ansatz(params)
        return qml.probs(wires=range(N))

    return energy, probs, const


def run_vqe_labs(
    N: int,
    layers: int = 3,
    n_steps: int = 150,
    stepsize: float = 0.05,
    seed: int = SEED,
) -> dict:
    energy, probs, const = make_vqe(N, layers)
    rng = np.random.default_rng(seed)
    params = pnp.array(rng.uniform(0, 2 * np.pi, size=(layers + 1, N)), requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    trace = []
    for _ in range(n_steps):
        params, e = opt.step_and_cost(energy, params)
        trace.append(float(e) + const)

    pr = np.asarray(probs(params))
    best_e, best_s = _decode_best(pr, N)
    return dict(strategy=f"VQE-HEA L={layers}", best_E=best_e, best_s=best_s,
                trace=trace, n_eval=n_steps)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_random_labs(N: int, n_eval: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    best_e, best_s = None, None
    trace = []
    for _ in range(n_eval):
        bits = rng.integers(0, 2, size=N)
        s = bits_to_spins(bits)
        e = labs_energy(s)
        if best_e is None or e < best_e:
            best_e, best_s = e, s
        trace.append(best_e)
    return dict(strategy="Random", best_E=best_e, best_s=best_s,
                trace=trace, n_eval=n_eval)


def run_sa_labs(N: int, n_steps: int = 5000, seed: int = SEED) -> dict:
    """Plain Metropolis SA on spins — dimod quadratization for N=20 is heavy."""
    rng = np.random.default_rng(seed)
    s = bits_to_spins(rng.integers(0, 2, size=N))
    e = labs_energy(s)
    best_e, best_s = e, s.copy()
    trace = [best_e]
    T0, T1 = 5.0, 0.01
    for step in range(n_steps):
        T = T0 * (T1 / T0) ** (step / n_steps)
        i = rng.integers(0, N)
        s_new = s.copy()
        s_new[i] *= -1
        e_new = labs_energy(s_new)
        if e_new < e or rng.random() < np.exp(-(e_new - e) / T):
            s, e = s_new, e_new
            if e < best_e:
                best_e, best_s = e, s.copy()
        trace.append(best_e)
    return dict(strategy="SA", best_E=best_e, best_s=best_s,
                trace=trace, n_eval=n_steps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def spin_str(s: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in s)


def report_row(N: int, row: dict, E_star: int) -> None:
    e = row["best_E"]
    f = N * N / (2 * e) if e > 0 else float("inf")
    r = f / (N * N / (2 * E_star))
    print(f"  {row['strategy']:<22} E={e:4d}  F={f:6.3f}  r={r:.3f}  N_eval={row['n_eval']}  s={spin_str(row['best_s'])}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--qaoa-p", type=int, default=2)
    p.add_argument("--vqe-layers", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=150)
    p.add_argument("--sa-steps", type=int, default=5000)
    p.add_argument("--random-eval", type=int, default=5000)
    p.add_argument("--skip-qaoa", action="store_true",
                   help="QAOA at N=20 is heavy; allow skipping for quick smoke tests")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    N = args.N
    E_star = 26 if N == 20 else None

    # (a)
    print("[Part a] Verifying LABS implementation")
    verify_barker11()

    # (b) — Strategies
    print(f"\n[Part b] N={N} strategies")
    rows = []

    if not args.skip_qaoa:
        t0 = time.perf_counter()
        rows.append(run_qaoa_labs(N, p=args.qaoa_p, n_steps=args.n_steps))
        print(f"  QAOA done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    rows.append(run_vqe_labs(N, layers=args.vqe_layers, n_steps=args.n_steps))
    print(f"  VQE done in {time.perf_counter()-t0:.1f}s")

    # (c) — Baselines
    print(f"\n[Part c] Baselines")
    t0 = time.perf_counter()
    rows.append(run_sa_labs(N, n_steps=args.sa_steps))
    print(f"  SA done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    rows.append(run_random_labs(N, n_eval=args.random_eval))
    print(f"  Random done in {time.perf_counter()-t0:.1f}s")

    # Report
    print(f"\n[Report] N={N}, target E*={E_star}")
    for row in rows:
        report_row(N, row, E_star or row["best_E"])

    # (d) discussion: TODO write 5-8 sentences in the report


if __name__ == "__main__":
    main()
