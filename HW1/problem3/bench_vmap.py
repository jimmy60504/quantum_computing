"""Benchmark: sequential QNode loop vs torch.func.vmap vs lightning.gpu.

Usage:
    python HW1/problem3/bench_vmap.py                         # 4q×2l (current)
    python HW1/problem3/bench_vmap.py --qubits 12 --layers 6 # larger circuit
    python HW1/problem3/bench_vmap.py --qubits 12 --layers 6 --gpu  # include GPU
"""

from __future__ import annotations

import argparse
import time

import pennylane as qml
import torch
from torch.func import vmap

BATCH_SIZE = 64
N_WARMUP = 2
N_TRIALS = 5
BATCHES_PER_EPOCH = 781


def make_circuit(num_qubits: int, num_layers: int, device_name: str, diff_method: str):
    dev = qml.device(device_name, wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)
        for layer in range(num_layers):
            for i in range(num_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit


def run_sequential(circuit, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        torch.stack(circuit(x[i], weights))
        for i in range(x.shape[0])
    ]).float()


def run_vmap(circuit, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    def circuit_tensor(inp, w):
        return torch.stack(circuit(inp, w)).float()

    batched = vmap(circuit_tensor, in_dims=(0, None), randomness="same")
    return batched(x, weights)


def bench(fn, x, weights, n_warmup, n_trials, label):
    for _ in range(n_warmup):
        w = weights.detach().clone().requires_grad_(True)
        out = fn(x, w)
        out.sum().backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(n_trials):
        w = weights.detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        out = fn(x, w)
        out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_t = sum(times) / len(times)
    print(f"  {label:<28}: {mean_t:.3f} s/batch  "
          f"({mean_t / BATCH_SIZE * 1000:.2f} ms/sample)  "
          f"→ epoch ≈ {mean_t * BATCHES_PER_EPOCH / 60:.1f} min")
    return mean_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu", action="store_true", help="Also benchmark lightning.gpu + adjoint")
    parser.add_argument("--gpu-only", action="store_true", help="Only benchmark GPU modes")
    args = parser.parse_args()

    q, l = args.qubits, args.layers
    print(f"\n{'='*62}")
    print(f"  Circuit: {q}q × {l}l   Batch: {args.batch}   "
          f"State-vector: 2^{q}={2**q} elements")
    print(f"{'='*62}")

    weights = torch.randn(l, q, 2)
    x = torch.randn(args.batch, q)

    results = {}

    if not args.gpu_only:
        cpu_circuit = make_circuit(q, l, "default.qubit", "backprop")
        print(f"\n  [CPU — default.qubit + backprop]")
        results["seq_cpu"] = bench(
            lambda x, w: run_sequential(cpu_circuit, x, w),
            x, weights, N_WARMUP, N_TRIALS, "sequential"
        )
        results["vmap_cpu"] = bench(
            lambda x, w: run_vmap(cpu_circuit, x, w),
            x, weights, N_WARMUP, N_TRIALS, "vmap (CPU)"
        )

    if args.gpu or args.gpu_only:
        try:
            gpu_circuit = make_circuit(q, l, "lightning.gpu", "adjoint")
            print(f"\n  [GPU — lightning.gpu + adjoint]")
            results["seq_gpu"] = bench(
                lambda x, w: run_sequential(gpu_circuit, x, w),
                x, weights, N_WARMUP, N_TRIALS, "sequential (GPU)"
            )
        except Exception as e:
            print(f"  lightning.gpu not available: {e}")

    print(f"\n  {'─'*55}")
    if "seq_cpu" in results and "vmap_cpu" in results:
        print(f"  vmap vs seq CPU:  {results['seq_cpu'] / results['vmap_cpu']:.1f}×  faster")
    if "seq_gpu" in results and "seq_cpu" in results:
        print(f"  GPU vs seq CPU:   {results['seq_cpu'] / results['seq_gpu']:.1f}×  faster")
    if "seq_gpu" in results and "vmap_cpu" in results:
        print(f"  vmap CPU vs GPU:  {results['seq_gpu'] / results['vmap_cpu']:.2f}×  "
              + ("← vmap wins" if results["vmap_cpu"] < results["seq_gpu"] else "← GPU wins"))
    print(f"  MLP reference:    ~0.004 s/batch  (~0.05 min/epoch)")
    print()


if __name__ == "__main__":
    main()
