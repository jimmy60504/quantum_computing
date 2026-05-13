"""gx10 runner for LABS Problem 3 — PCE + Hybrid VQE benchmark.

Runs all strategies for N=20 and writes results to results_gx10.json.
Usage (inside container):
    python3 run_gx10.py [--budget 5000] [--n-steps 40] [--pce-backend torch]
"""
from __future__ import annotations

import argparse
import json
import time
import sys
import os

# Make sure solution.py is importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from solution import (
    N_DEFAULT, SEED,
    verify_barker11,
    run_vqe_labs,
    run_hybrid_vqe_labs,
    run_pce_labs,
    run_sa_labs,
    run_random_labs,
    spin_str,
)


def report(N: int, row: dict, E_star: int = 26) -> str:
    e = row["best_E"]
    f = N * N / (2 * e) if e > 0 else float("inf")
    r = f / (N * N / (2 * E_star))
    s = spin_str(row["best_s"])
    return (
        f"  {row['strategy']:<25} E={e:4d}  F={f:6.3f}"
        f"  r={r:.3f}  N_eval={row['n_eval']}  s={s}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--budget", type=int, default=5000)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--sa-steps", type=int, default=800)
    ap.add_argument("--baseline-sa-steps", type=int, default=50000)
    ap.add_argument("--vqe-layers", type=int, default=4)
    ap.add_argument("--pce-m", type=int, default=4)
    ap.add_argument("--pce-layers", type=int, default=4)
    ap.add_argument("--pce-backend", choices=["torch", "pennylane"], default="torch")
    ap.add_argument("--pce-ansatz", choices=["rxryrz", "rot", "ry"], default="rxryrz")
    ap.add_argument("--pce-decode-every", type=int, default=10)
    ap.add_argument("--pce-batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=11224003)
    ap.add_argument("--skip-vqe", action="store_true")
    ap.add_argument("--skip-hybrid", action="store_true")
    ap.add_argument("--skip-pce", action="store_true")
    args = ap.parse_args()

    N = args.N
    print(f"[LABS gx10] N={N}, budget={args.budget}, n_steps={args.n_steps}")
    verify_barker11()

    rows = []

    if not args.skip_vqe:
        print("\n--- VQE-HEA ---")
        t0 = time.perf_counter()
        r = run_vqe_labs(N, layers=args.vqe_layers, n_steps=args.n_steps)
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print(report(N, r))
        rows.append(r)

    if not args.skip_hybrid:
        print("\n--- Hybrid VQE + SA ---")
        t0 = time.perf_counter()
        r = run_hybrid_vqe_labs(
            N, layers=args.vqe_layers,
            n_steps=args.n_steps,
            budget=args.budget,
            sa_steps=args.sa_steps,
        )
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print(report(N, r))
        rows.append(r)

    if not args.skip_pce:
        print("\n--- PCE VQE (Sciorilli et al. 2025) ---")
        t0 = time.perf_counter()
        if args.pce_backend == "torch":
            from pce_torch import run_pce_labs_torch

            r = run_pce_labs_torch(
                N,
                m=args.pce_m,
                layers=args.pce_layers,
                n_steps=args.n_steps,
                budget=args.budget,
                batch_size=args.pce_batch_size,
                seed=args.seed,
                decode_every=args.pce_decode_every,
            )
        else:
            r = run_pce_labs(
                N, m=args.pce_m,
                layers=args.pce_layers,
                n_steps=args.n_steps,
                budget=args.budget,
                sa_steps=args.sa_steps,
                seed=args.seed,
                ansatz_mode=args.pce_ansatz,
                decode_every=args.pce_decode_every,
            )
        print(f"  done in {time.perf_counter()-t0:.1f}s")
        print(report(N, r))
        rows.append(r)

    print("\n--- SA baseline ---")
    t0 = time.perf_counter()
    r = run_sa_labs(N, n_steps=args.baseline_sa_steps)
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    print(report(N, r))
    rows.append(r)

    # Serialise results
    out = []
    E_star = 26
    for row in rows:
        e = row["best_E"]
        f = N * N / (2 * e)
        rr = f / (N * N / (2 * E_star))
        out.append({
            "strategy": row["strategy"],
            "best_E": e,
            "best_F": round(f, 4),
            "r": round(rr, 4),
            "n_eval": row["n_eval"],
            "best_s": spin_str(row["best_s"]),
            "trace": row["trace"],
        })

    outfile = os.path.join(os.path.dirname(__file__), "results_gx10.json")
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nResults saved → {outfile}")


if __name__ == "__main__":
    main()
