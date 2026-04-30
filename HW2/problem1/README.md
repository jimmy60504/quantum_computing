# HW2 Problem 1 — 01 Knapsack: QUBO and Quantum Annealing

10-item knapsack with W = 165. Formulate as QUBO and solve with D-Wave's Ocean SDK.

| Item   | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 |
|--------|----|----|----|----|----|----|----|----|----|----|
| Weight | 23 | 31 | 29 | 44 | 53 | 38 | 63 | 85 | 89 | 82 |
| Value  | 92 | 57 | 49 | 68 | 60 | 43 | 67 | 84 | 87 | 72 |

## Sub-tasks

**(a)** Classical solution (brute-force or DP): optimal items, total weight, total value.  
Derive the QUBO formulation (handle inequality via slack variables or unbalanced penalization); write out Q matrix entries.

**(b)** Solve QUBO with `dimod.ExactSolver`. Investigate penalty coefficient λ (≥ 3 values); report feasibility and optimality. Heuristic: λ > max(v_i) / min(w_i) = 92/23 ≈ 4.

**(c)** Solve with `neal.SimulatedAnnealingSampler`. Sweep `num_reads` ∈ {10, 100, 1000, 10000}. Report success probability (fraction of reads returning optimal bitstring from (b)).

**(d)** Comparison table: method, best value, total weight, feasibility, computation time. Brief discussion (3–5 sentences).

## Key formulas

Slack-variable QUBO (M = ⌈log₂(165)⌉ = 8 slack bits, 18 total variables):

```
min_{x,s} [ -∑ v_i x_i  +  λ (∑ w_i x_i + ∑ 2^k s_k − W)² ]
```
