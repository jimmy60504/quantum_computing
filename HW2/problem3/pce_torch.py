"""Batched PyTorch statevector backend for the HW2 LABS PCE experiment.

This is meant for gx10-style execution.  It trains many independent 4-qubit
PCE initializations in parallel, which is the practical way to reproduce the
paper's many-random-restart protocol without waiting on one PennyLane QNode at
a time.
"""

from __future__ import annotations

import numpy as np

from solution import (
    SEED,
    _all_pauli_strings,
    _balanced_pauli_assignment,
    _max_noncommuting_assignment,
)


def _spin_str(s: np.ndarray) -> str:
    return "".join("+" if x > 0 else "-" for x in s)


def _pauli_assignment(N: int, m: int, mode: str) -> list[str]:
    if mode == "nc":
        return _max_noncommuting_assignment(N, m)
    if mode == "balanced":
        return _balanced_pauli_assignment(N, m)
    if mode == "lex":
        return _all_pauli_strings(m)[:N]
    raise ValueError(f"unknown pauli_mode={mode!r}")


def run_pce_labs_torch(
    N: int,
    m: int = 4,
    layers: int = 4,
    budget: int = 5000,
    batch_size: int = 128,
    n_steps: int = 40,
    stepsize: float = 0.035,
    seed: int = SEED + 2,
    pauli_mode: str = "nc",
    alpha: float | None = None,
    beta: float = 15.0,
    decode_every: int = 10,
    device: str | None = None,
) -> dict:
    """Run PCE with batched random restarts on a statevector simulator.

    `n_eval` counts circuit-state evaluations as `batch_size` per optimizer
    step.  Decode checks do not add to this count, matching the convention used
    by the rest of this homework script where optimizer cost calls are the main
    budget proxy.
    """
    import torch

    alpha = alpha if alpha is not None else 1.5 * m
    decode_every = max(1, decode_every)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cdtype = torch.complex64
    rdtype = torch.float32
    dim = 2**m

    paulis = _pauli_assignment(N, m, pauli_mode)

    eye = torch.eye(2, dtype=cdtype)
    pauli_mats = {
        "I": eye,
        "X": torch.tensor([[0, 1], [1, 0]], dtype=cdtype),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=cdtype),
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=cdtype),
    }

    def kron_all(label: str):
        out = pauli_mats[label[0]]
        for char in label[1:]:
            out = torch.kron(out, pauli_mats[char])
        return out

    obs = torch.stack([kron_all(p) for p in paulis]).to(device)

    def cnot_source_for_output(control: int, target: int):
        source = np.zeros(dim, dtype=np.int64)
        for old in range(dim):
            bits = [(old >> (m - 1 - i)) & 1 for i in range(m)]
            new_bits = bits[:]
            if bits[control]:
                new_bits[target] ^= 1
            new = 0
            for bit in new_bits:
                new = (new << 1) | bit
            source[new] = old
        return torch.tensor(source, device=device)

    cnot_sources = [cnot_source_for_output(w, (w + 1) % m) for w in range(m)]

    def apply_1q(state, gate, wire: int):
        current_batch = state.shape[0]
        shaped = state.view(current_batch, *([2] * m))
        shaped = torch.movedim(shaped, wire + 1, -1).contiguous().view(current_batch, -1, 2)
        shaped = torch.einsum("bij,brj->bri", gate, shaped)
        shaped = shaped.view(current_batch, *([2] * (m - 1)), 2)
        return torch.movedim(shaped, -1, wire + 1).contiguous().view(current_batch, dim)

    def rx(theta):
        c = torch.cos(theta / 2).to(cdtype)
        s = (-1j * torch.sin(theta / 2)).to(cdtype)
        gate = torch.zeros(theta.shape[0], 2, 2, device=device, dtype=cdtype)
        gate[:, 0, 0] = c
        gate[:, 1, 1] = c
        gate[:, 0, 1] = s
        gate[:, 1, 0] = s
        return gate

    def ry(theta):
        c = torch.cos(theta / 2).to(cdtype)
        s = torch.sin(theta / 2).to(cdtype)
        gate = torch.zeros(theta.shape[0], 2, 2, device=device, dtype=cdtype)
        gate[:, 0, 0] = c
        gate[:, 1, 1] = c
        gate[:, 0, 1] = -s
        gate[:, 1, 0] = s
        return gate

    def rz(theta):
        a = torch.exp((-0.5j * theta).to(cdtype))
        b = torch.exp((0.5j * theta).to(cdtype))
        gate = torch.zeros(theta.shape[0], 2, 2, device=device, dtype=cdtype)
        gate[:, 0, 0] = a
        gate[:, 1, 1] = b
        return gate

    def circuit(params):
        current_batch = params.shape[0]
        state = torch.zeros(current_batch, dim, device=device, dtype=cdtype)
        state[:, 0] = 1 + 0j
        for layer in range(layers + 1):
            for wire in range(m):
                state = apply_1q(state, rx(params[:, layer, wire, 0]), wire)
                state = apply_1q(state, ry(params[:, layer, wire, 1]), wire)
                state = apply_1q(state, rz(params[:, layer, wire, 2]), wire)
            if layer < layers:
                for source in cnot_sources:
                    state = state[:, source]
        return state

    def labs_energy_batch(signs):
        energy = torch.zeros(signs.shape[0], device=device)
        for k in range(1, N):
            corr = (signs[:, : N - k] * signs[:, k:]).sum(dim=1)
            energy = energy + corr * corr
        return energy

    best_e: int | None = None
    best_s: np.ndarray | None = None
    trace: list[int] = []
    total_evals = 0
    batch_index = 0

    while total_evals + batch_size <= budget:
        steps_this_batch = min(n_steps, (budget - total_evals) // batch_size)
        if steps_this_batch <= 0:
            break

        torch.manual_seed(seed + batch_index)
        params = torch.nn.Parameter(
            2 * np.pi * torch.rand(
                batch_size,
                layers + 1,
                m,
                3,
                device=device,
                dtype=rdtype,
            )
        )
        optimizer = torch.optim.Adam([params], lr=stepsize)

        for step in range(steps_this_batch + 1):
            if step > 0:
                optimizer.zero_grad(set_to_none=True)
                psi = circuit(params)
                raw = torch.einsum("bi,nij,bj->bn", psi.conj(), obs, psi).real
                x = torch.tanh(alpha * raw)
                relaxed_energy = torch.zeros(batch_size, device=device)
                for k in range(1, N):
                    corr = (x[:, : N - k] * x[:, k:]).sum(dim=1)
                    relaxed_energy = relaxed_energy + corr * corr
                loss = (relaxed_energy - beta * (x * x).sum(dim=1)).mean()
                loss.backward()
                optimizer.step()
                total_evals += batch_size

            if step % decode_every == 0 or step == steps_this_batch:
                with torch.no_grad():
                    psi = circuit(params)
                    raw = torch.einsum("bi,nij,bj->bn", psi.conj(), obs, psi).real
                    signs = torch.where(raw >= 0, torch.ones_like(raw), -torch.ones_like(raw))
                    discrete_energy = labs_energy_batch(signs)
                    value, index = discrete_energy.min(dim=0)
                    candidate_e = int(value.item())
                    if best_e is None or candidate_e < best_e:
                        best_e = candidate_e
                        best_s = signs[index].detach().cpu().numpy().astype(int)
                    trace.append(best_e)
                    if best_e == 26:
                        return {
                            "strategy": f"PCE-torch m={m} L={layers} rxryrz",
                            "best_E": best_e,
                            "best_s": best_s,
                            "trace": trace,
                            "n_eval": total_evals,
                            "device": device,
                            "batch_size": batch_size,
                        }

        batch_index += 1

    return {
        "strategy": f"PCE-torch m={m} L={layers} rxryrz",
        "best_E": best_e,
        "best_s": best_s,
        "trace": trace,
        "n_eval": total_evals,
        "device": device,
        "batch_size": batch_size,
    }


if __name__ == "__main__":
    result = run_pce_labs_torch()
    print(
        f"{result['strategy']} E={result['best_E']} "
        f"N_eval={result['n_eval']} s={_spin_str(result['best_s'])}"
    )
