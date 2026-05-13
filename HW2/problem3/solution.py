"""HW2 Problem 3 — LABS for N=20.

Strategies implemented:
    1. Quartic-Hamiltonian QAOA  (PennyLane, depth p)
    2. VQE with hardware-efficient ansatz (RY layers + ring CNOTs)
    3. Hybrid VQE: multi-restart VQE + SA polish on sampled states
    4. PCE VQE: Pauli Correlation Encoding (Sciorilli et al. 2025)
       — encodes N spins as expectation values of N Pauli strings on m≪N qubits

Baselines:
    - Random sampling under the same shot budget
    - Simulated annealing (neal) on a quadratized QUBO
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Problem data / cost function
# ---------------------------------------------------------------------------

SEED = 11224001
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
    dev = qml.device("lightning.qubit", wires=N)

    def circuit(params):
        g, b = params[0], params[1]
        for w in range(N):
            qml.Hadamard(wires=w)
        for layer in range(p):
            # H_C terms all commute (Pauli-Z products) → n=1 is exact
            qml.ApproxTimeEvolution(H_C, g[layer], 1)
            qml.ApproxTimeEvolution(H_M, b[layer], 1)

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
    dev = qml.device("lightning.qubit", wires=N)

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
# Strategy 3 — Hybrid: multi-restart VQE + classical local search
# ---------------------------------------------------------------------------

def _local_search(s_init: np.ndarray, n_steps: int, rng) -> tuple[int, np.ndarray]:
    """Metropolis SA starting from s_init.  Classical — free in N_eval.

    T0 is calibrated from the local 1-flip neighbourhood so that the initial
    acceptance rate is ~50 %, allowing escape from deep local minima.
    """
    s = s_init.copy()
    e = labs_energy(s)
    best_e, best_s = e, s.copy()
    N = len(s)

    # Calibrate T0: median ΔE of 1-flip neighbours
    delta_es = []
    for i in range(N):
        s[i] *= -1
        delta_es.append(abs(labs_energy(s) - e))
        s[i] *= -1
    median_delta = float(np.median(delta_es))
    # Set T0 so P(accept median ΔE) ≈ 0.5  →  T0 = median_ΔE / ln(2)
    T0 = max(median_delta / np.log(2), 1.0)
    T1 = 0.01

    for step in range(n_steps):
        T = T0 * (T1 / T0) ** (step / n_steps)
        i = int(rng.integers(0, N))
        s[i] *= -1
        e_new = labs_energy(s)
        if e_new < e or rng.random() < np.exp(-(e_new - e) / T):
            e = e_new
            if e < best_e:
                best_e, best_s = e, s.copy()
        else:
            s[i] *= -1          # revert
    return best_e, best_s


def run_hybrid_vqe_labs(
    N: int,
    layers: int = 4,
    n_steps: int = 200,
    budget: int = 5000,
    top_k: int = 128,
    sa_steps: int = 600,
    stepsize: float = 0.05,
    seed: int = SEED,
) -> dict:
    """Multi-restart VQE with classical SA polish on sampled states.

    Circuit eval budget = n_restarts × n_steps  (≤ budget).
    After each VQE run the top-K basis states by probability are polished
    with a short SA run (classical → free N_eval).
    """
    energy_fn, probs_fn, const = make_vqe(N, layers)
    rng = np.random.default_rng(seed)

    best_e: int | None = None
    best_s: np.ndarray | None = None
    all_trace: list[float] = []
    total_evals = 0

    while total_evals + n_steps <= budget:
        # ── fresh random init ──────────────────────────────────────────────
        params = pnp.array(
            rng.uniform(0, 2 * np.pi, size=(layers + 1, N)),
            requires_grad=True,
        )
        opt = qml.AdamOptimizer(stepsize=stepsize)
        for _ in range(n_steps):
            params, e = opt.step_and_cost(energy_fn, params)
            all_trace.append(float(e) + const)
        total_evals += n_steps

        # ── decode: top-K states by probability ───────────────────────────
        pr = np.asarray(probs_fn(params))
        K = min(top_k, len(pr))
        top_idx = np.argsort(pr)[-K:]

        # ── classical SA polish on each candidate ─────────────────────────
        for idx in top_idx:
            bits = np.array([(idx >> (N - 1 - i)) & 1 for i in range(N)])
            s_init = bits_to_spins(bits)
            e_pol, s_pol = _local_search(s_init, sa_steps, rng)
            if best_e is None or e_pol < best_e:
                best_e, best_s = e_pol, s_pol.copy()

        if best_e == 26:          # optimal found — stop early
            break

    return dict(
        strategy=f"Hybrid-VQE L={layers}",
        best_E=best_e,
        best_s=best_s,
        trace=all_trace,
        n_eval=total_evals,
    )


# ---------------------------------------------------------------------------
# Strategy 4 — PCE VQE (Pauli Correlation Encoding)
# Reference: Sciorilli et al., arXiv:2506.17391 (2025)
#
# Key details from the paper:
#   1. tanh relaxation:  x̃ᵢ = tanh(α⟨Πᵢ⟩),  α = 1.5·m
#      pushes expectation values towards ±1 (discrete-like)
#   2. Regularisation:   ℒ = Σ_ℓ Cℓ² - β Σᵢ x̃ᵢ²,  β = 15
#      prevents trivial collapse to x̃ᵢ = 0
#   3. Maximally non-commuting Pauli assignment (Π^NC):
#      greedily pick Paulis that anti-commute with the most already-selected
#   4. Complex single-qubit rotations are important: an RY-only real ansatz
#      makes odd-Y Pauli expectations identically zero and freezes bits.
# ---------------------------------------------------------------------------

def _all_pauli_strings(m: int) -> list[str]:
    """All 4^m − 1 non-identity Pauli strings on m qubits (lex order)."""
    return [
        "".join(combo)
        for combo in itertools.product("IXYZ", repeat=m)
        if any(c != "I" for c in combo)
    ]


def _do_anticommute(p: str, q: str) -> bool:
    """Return True iff Pauli strings p and q anti-commute."""
    # Two Pauli strings anti-commute iff the number of positions where
    # both are non-I and they differ (i.e. {X,Y}, {Y,Z}, or {X,Z} pairs)
    # is odd.
    count = 0
    for a, b in zip(p, q):
        if a != "I" and b != "I" and a != b:
            count += 1
    return count % 2 == 1


def _max_noncommuting_assignment(N: int, m: int) -> list[str]:
    """Greedily select N Paulis that maximise mutual anti-commutativity.

    At each step we pick the candidate that anti-commutes with the most
    already-selected Paulis (ties broken by lexicographic order).
    This is the Π^(NC) construction from Sciorilli et al. 2025.
    """
    candidates = _all_pauli_strings(m)
    selected: list[str] = [candidates[0]]   # seed with first Pauli
    remaining = candidates[1:]

    while len(selected) < N:
        # Count anti-commutations with already selected set
        scores = {
            p: sum(1 for s in selected if _do_anticommute(p, s))
            for p in remaining
        }
        best = max(remaining, key=lambda p: (scores[p], p))
        selected.append(best)
        remaining.remove(best)

    return selected


def _balanced_pauli_assignment(N: int, m: int) -> list[str]:
    """Select N Pauli strings that spread information across all m qubits.

    Strategy: prefer weight-2 Paulis (exactly 2 non-I positions) because
    they activate all qubits evenly.  Pad with weight-1 or weight-3 if needed.
    Sorts weight-2 pool so qubit pairs appear in round-robin order.
    """
    by_weight: dict[int, list[str]] = {}
    for combo in itertools.product("IXYZ", repeat=m):
        w = sum(1 for c in combo if c != "I")
        if w == 0:
            continue
        by_weight.setdefault(w, []).append("".join(combo))

    # Round-robin over qubit pairs so all qubits appear equally
    pairs = list(itertools.combinations(range(m), 2))
    pool: list[str] = []
    for nz1, nz2 in itertools.cycle(pairs):
        batch = [
            p for p in by_weight.get(2, [])
            if p[nz1] != "I" and p[nz2] != "I"
            and all(p[j] == "I" for j in range(m) if j != nz1 and j != nz2)
            and p not in pool
        ]
        pool.extend(batch)
        if len(pool) >= N:
            break

    # Pad with weight-1, then weight-3 if still short
    for w in (1, 3, 4):
        for p in by_weight.get(w, []):
            if p not in pool:
                pool.append(p)
            if len(pool) >= N:
                break
        if len(pool) >= N:
            break

    return pool[:N]


def _pauli_str_to_op(s: str) -> qml.operation.Operator:
    """'XZIY' → PennyLane tensor-product operator on wires 0,1,…"""
    ops = []
    for i, c in enumerate(s):
        if c == "X":
            ops.append(qml.PauliX(i))
        elif c == "Y":
            ops.append(qml.PauliY(i))
        elif c == "Z":
            ops.append(qml.PauliZ(i))
    if not ops:
        return qml.Identity(0)
    result = ops[0]
    for op in ops[1:]:
        result = result @ op
    return result


def _pce_param_shape(layers: int, m: int, ansatz_mode: str) -> tuple[int, ...]:
    if ansatz_mode == "ry":
        return (layers + 1, m)
    if ansatz_mode in {"rxryrz", "rot"}:
        return (layers + 1, m, 3)
    raise ValueError(f"unknown PCE ansatz_mode={ansatz_mode!r}")


def make_pce(
    N: int,
    m: int = 4,
    layers: int = 4,
    pauli_mode: str = "nc",   # "nc" | "balanced" | "lex"
    ansatz_mode: str = "rxryrz",
    alpha: float | None = None,
    beta: float = 15.0,
):
    """Build PCE energy / decode functions for LABS on N spins using m qubits.

    Implements the full Sciorilli et al. 2025 recipe:
      • tanh relaxation:   x̃ᵢ = tanh(α⟨Πᵢ⟩),  α = 1.5·m  (default)
      • Regularisation:    ℒ = Σ_k Cₖ² − β Σᵢ x̃ᵢ²,  β = 15
      • Pauli assignment:  "nc" = maximally non-commuting (Π^NC, best)
                           "balanced" = weight-2 round-robin
                           "lex" = naive lex order (worst)
      • Ansatz:            "rxryrz" = complex hardware-efficient layers
                           "ry" = legacy real-valued ansatz
    """
    alpha = alpha if alpha is not None else 1.5 * m
    ansatz_mode = ansatz_mode.lower()
    _pce_param_shape(layers, m, ansatz_mode)  # validate early

    if pauli_mode == "nc":
        pauli_assign = _max_noncommuting_assignment(N, m)
    elif pauli_mode == "balanced":
        pauli_assign = _balanced_pauli_assignment(N, m)
    else:
        pauli_assign = _all_pauli_strings(m)[:N]
    assert len(pauli_assign) == N

    obs_list = [_pauli_str_to_op(p) for p in pauli_assign]
    dev = qml.device("lightning.qubit", wires=m)

    def _ansatz(params):
        # "rxryrz" follows the paper's RX/RY/RZ single-qubit expressivity.
        # The "ry" path is kept for ablations and backwards comparison.
        if ansatz_mode == "ry":
            for w in range(m):
                qml.RY(params[0, w], wires=w)
        elif ansatz_mode == "rot":
            for w in range(m):
                qml.Rot(params[0, w, 0], params[0, w, 1], params[0, w, 2], wires=w)
        else:
            for w in range(m):
                qml.RX(params[0, w, 0], wires=w)
                qml.RY(params[0, w, 1], wires=w)
                qml.RZ(params[0, w, 2], wires=w)
        for lay in range(layers):
            for w in range(m):
                qml.CNOT(wires=[w, (w + 1) % m])
            if ansatz_mode == "ry":
                for w in range(m):
                    qml.RY(params[lay + 1, w], wires=w)
            elif ansatz_mode == "rot":
                for w in range(m):
                    qml.Rot(
                        params[lay + 1, w, 0],
                        params[lay + 1, w, 1],
                        params[lay + 1, w, 2],
                        wires=w,
                    )
            else:
                for w in range(m):
                    qml.RX(params[lay + 1, w, 0], wires=w)
                    qml.RY(params[lay + 1, w, 1], wires=w)
                    qml.RZ(params[lay + 1, w, 2], wires=w)

    @qml.qnode(dev, interface="autograd")
    def _spin_exp(params):
        _ansatz(params)
        return [qml.expval(obs) for obs in obs_list]

    def energy(params):
        """PCE cost with tanh relaxation and regularisation (Sciorilli 2025).

        ℒ = Σ_k (Σᵢ x̃ᵢ x̃ᵢ₊ₖ)² − β Σᵢ x̃ᵢ²
        where x̃ᵢ = tanh(α ⟨Πᵢ⟩).
        """
        raw = pnp.stack(_spin_exp(params))      # shape (N,)  raw ⟨Πᵢ⟩
        x = pnp.tanh(alpha * raw)               # tanh relaxation → ±1
        E = pnp.array(0.0)
        for k in range(1, N):
            Ck = pnp.dot(x[: N - k], x[k:])
            E = E + Ck * Ck
        reg = beta * pnp.sum(x ** 2)            # regularisation
        return E - reg

    def decode(params) -> np.ndarray:
        """Map continuous expectation values → discrete ±1 spin sequence."""
        raw = np.array([float(v) for v in _spin_exp(params)])
        return np.where(raw >= 0, +1, -1).astype(int)

    return energy, decode


def run_pce_labs(
    N: int,
    m: int = 4,
    layers: int = 4,
    n_steps: int = 200,
    budget: int = 5000,
    sa_steps: int = 800,
    stepsize: float = 0.05,
    seed: int = SEED,
    pauli_mode: str = "nc",
    ansatz_mode: str = "rxryrz",
    decode_every: int = 10,
    beta: float = 15.0,
) -> dict:
    """Multi-restart PCE VQE with SA polish on decoded sequences.

    The quantum circuit has only m qubits regardless of N.
    N_eval counts gradient circuit evaluations; SA is classical (free).
    Uses the full Sciorilli et al. 2025 recipe (tanh + β regularisation).
    Discrete LABS energy is checked during training, since a good rounded
    sequence can appear before the final variational parameters.
    """
    ansatz_mode = ansatz_mode.lower()
    param_shape = _pce_param_shape(layers, m, ansatz_mode)
    energy_fn, decode_fn = make_pce(
        N,
        m,
        layers,
        pauli_mode=pauli_mode,
        ansatz_mode=ansatz_mode,
        beta=beta,
    )
    rng = np.random.default_rng(seed)

    best_e: int | None = None
    best_s: np.ndarray | None = None
    all_trace: list[float] = []
    total_evals = 0
    decode_every = max(1, decode_every)

    while total_evals < budget:
        steps_this_restart = min(n_steps, budget - total_evals)
        params = pnp.array(
            rng.uniform(0, 2 * np.pi, size=param_shape),
            requires_grad=True,
        )
        opt = qml.AdamOptimizer(stepsize=stepsize)
        local_best_s: np.ndarray | None = None
        local_best_e: int | None = None

        for step in range(steps_this_restart):
            params, e = opt.step_and_cost(energy_fn, params)
            all_trace.append(float(e))
            total_evals += 1

            if (step + 1) % decode_every == 0 or step == steps_this_restart - 1:
                s_dec = decode_fn(params)
                e_dec = labs_energy(s_dec)
                if local_best_e is None or e_dec < local_best_e:
                    local_best_e, local_best_s = e_dec, s_dec.copy()
                if best_e is None or e_dec < best_e:
                    best_e, best_s = e_dec, s_dec.copy()
                if best_e == 26:
                    break

        if local_best_s is None:
            local_best_s = decode_fn(params)
            local_best_e = labs_energy(local_best_s)
            if best_e is None or local_best_e < best_e:
                best_e, best_s = local_best_e, local_best_s.copy()

        if sa_steps > 0:
            e_pol, s_pol = _local_search(local_best_s, sa_steps, rng)
            if best_e is None or e_pol < best_e:
                best_e, best_s = e_pol, s_pol.copy()
        if best_e == 26:
            break

    return dict(
        strategy=f"PCE m={m} L={layers} {ansatz_mode}",
        best_E=best_e,
        best_s=best_s,
        trace=all_trace,
        n_eval=total_evals,
    )


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


def _calibrated_temperature(s: np.ndarray) -> float:
    """Choose T0 from the local 1-flip energy scale.

    LABS has tall local barriers even at N=20.  A fixed small temperature makes
    Metropolis search look artificially weak, so we set the initial temperature
    such that a median uphill 1-flip move is accepted with probability ~1/2.
    """
    e = labs_energy(s)
    delta_es = []
    for i in range(len(s)):
        s[i] *= -1
        delta_es.append(abs(labs_energy(s) - e))
        s[i] *= -1
    median_delta = float(np.median(delta_es))
    return max(median_delta / np.log(2), 1.0)


def run_sa_labs(
    N: int,
    n_steps: int = 5000,
    seed: int = SEED,
    restart_steps: int = 10000,
) -> dict:
    """Multi-restart Metropolis SA with a locally calibrated temperature scale."""
    rng = np.random.default_rng(seed)
    best_e: int | None = None
    best_s: np.ndarray | None = None
    trace: list[int] = []
    total_steps = 0

    while total_steps < n_steps:
        steps_this_restart = min(restart_steps, n_steps - total_steps)
        s = bits_to_spins(rng.integers(0, 2, size=N))
        e = labs_energy(s)
        if best_e is None or e < best_e:
            best_e, best_s = e, s.copy()

        T0 = _calibrated_temperature(s)
        T1 = 0.01
        for step in range(steps_this_restart):
            T = T0 * (T1 / T0) ** (step / max(steps_this_restart, 1))
            i = int(rng.integers(0, N))
            s[i] *= -1
            e_new = labs_energy(s)
            if e_new < e or rng.random() < np.exp(-(e_new - e) / T):
                e = e_new
                if e < best_e:
                    best_e, best_s = e, s.copy()
            else:
                s[i] *= -1
            trace.append(best_e)
            total_steps += 1

        if best_e == 26:
            break

    return dict(strategy="SA-calibrated", best_E=best_e, best_s=best_s,
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
    p.add_argument("--pce-layers", type=int, default=4)
    p.add_argument("--pce-ansatz", choices=["rxryrz", "rot", "ry"], default="rxryrz")
    p.add_argument("--pce-decode-every", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=150)
    p.add_argument("--sa-steps", type=int, default=50000)
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

    t0 = time.perf_counter()
    rows.append(run_hybrid_vqe_labs(N, layers=args.vqe_layers, n_steps=args.n_steps))
    print(f"  Hybrid-VQE done in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    rows.append(
        run_pce_labs(
            N,
            m=4,
            layers=args.pce_layers,
            n_steps=args.n_steps,
            ansatz_mode=args.pce_ansatz,
            decode_every=args.pce_decode_every,
        )
    )
    print(f"  PCE done in {time.perf_counter()-t0:.1f}s")

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
