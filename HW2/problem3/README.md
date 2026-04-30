# HW2 Problem 3 — Low Autocorrelation Binary Sequences (LABS)

Updated from `QCAA_HW2_updated0421.pdf`.

Find a binary sequence `s = (s_1, ..., s_N)`, with `s_i ∈ {-1, +1}`, that
**maximizes** the merit factor `F(s) = N^2 / (2E(s))`, equivalently
**minimizes** the sidelobe energy

```text
C_k(s) = sum_{i=1}^{N-k} s_i s_{i+k}
E(s)   = sum_{k=1}^{N-1} C_k(s)^2
F(s)   = N^2 / (2E(s))
```

LABS is harder than Max-Cut because the expanded cost has nonlocal 2-spin and
4-spin interactions, and the landscape is glassy with exponentially many local
minima.

## Benchmark Target

Solve LABS for `N = 20`.

- Known optimum: `E* = 26`
- Optimal merit factor: `F* = 400 / 52 ≈ 7.692`
- Merit factor ratio: `r = F_best / F*`
- Budget: `N_eval` is the total number of quantum circuit evaluations across
  the entire hybrid pipeline
- Goal: `r >= 0.85` using `N_eval <= 5000`

For `N = 20`, the threshold effectively requires finding an optimal sequence
with `E = 26`, since the next energy level `E = 34` has only `r ≈ 0.765`.

## Verification Instances

Use these small cases before running the `N = 20` benchmark.

| N  | Sequence      | E* | F*     |
|----|---------------|----|--------|
| 5  | `+++-+`       | 2  | 6.25   |
| 11 | `+++---+--+-` | 5  | 12.10  |
| 13 | Barker        | 6  | 14.083 |

Important correction from the updated handout: the `N = 11` Barker sequence has
`E = 5` and `F = 12.10` under this convention.

## Sub-tasks

**(a)** Implement the LABS cost function `E(s)` and merit factor `F(s)`. Verify
the implementation on the `N = 11` Barker sequence `+++---+--+-`, whose
autocorrelations give `E = 5` and `F = 12.10`. State which two or more
strategies you chose and how LABS is encoded into a Hamiltonian, either quartic
or quadratized.

**(b)** Implement each strategy and apply it to LABS with `N = 20`. Report the
best sequence found, `E_best`, `F_best`, `r`, and `N_eval` for each.

**(c)** Provide a comparison table of your strategies alongside two baselines:
random sampling with the same total shot budget, and a purely classical
baseline such as simulated annealing or simple Tabu search. Plot convergence
curves, cost vs. iteration, for each strategy on the same axes.

**(d)** Write a discussion of 5-8 sentences: which strategy works best on LABS
and why? How does the glassy, quartic LABS landscape differ from the quadratic
Max-Cut landscape of Problem 2? What design choices have the largest impact?
Relate the findings to Ref. [4] or Ref. [6].

## Suggested Strategies

Implement at least two. You may also propose your own.

- **Quartic-Hamiltonian QAOA**: build `H_C` as Pauli-Z products of degree up to
  4 and compare depths `p`.
- **Quadratization**: introduce auxiliary variables for spin products and
  enforce them with 2-local QUBO gadgets, then solve with QAOA, VQE, or quantum
  annealing.
- **VQE**: use a hardware-efficient ansatz, such as layered `RY` rotations with
  CNOT entanglers, and optimize the expectation value of `H_C^LABS`.
- **Quantum annealing (simulated)**: quadratize LABS and solve with D-Wave
  samplers as in Problem 1.
- **Warm-starting / INTERP / RQAOA**: apply advanced QAOA techniques from the
  handout appendix.
- **Classical optimizer comparison**: compare at least three classical
  optimizers for the variational outer loop, such as gradient descent, COBYLA,
  Nelder-Mead, SPSA, or Adam.
- **Hybrid quantum-classical search**: use quantum samples from QAOA, VQE, or
  annealing to seed classical local search, then feed improved solutions back
  into the next quantum iteration.
- **Novel approach**: any other quantum or hybrid strategy with a non-trivial
  quantum component.

## Quartic Hamiltonian Notes

Expanding the sidelobe energy gives terms with degree at most 4:

```text
E(s) = sum_{k=1}^{N-1} sum_{i,j=1}^{N-k} s_i s_{i+k} s_j s_{j+k}
```

Under `s_i -> Z_i`, the LABS cost Hamiltonian is

```text
H_C^LABS = sum_{k=1}^{N-1} sum_{i,j=1}^{N-k}
           Z_i Z_{i+k} Z_j Z_{j+k}
```

Simplify repeated Pauli factors before implementation. Terms with `i = j`
reduce to identity, overlapping pairs can reduce to 2-local terms, and identity
terms add the constant `N(N - 1) / 2` to `E`. The constant may be dropped during
optimization, but it must be restored when reporting `E` and `F`.

## Quadratization Notes

One route is to introduce auxiliary variables `y_ij = s_i s_j` so that each
`C_k^2` becomes quadratic in product variables. The constraints `y_ij = s_i s_j`
must be converted to valid 2-local QUBO terms, for example by converting spins
to binary variables and using product/XOR gadgets or a library quadratizer. For
`N = 20`, this can require roughly `O(N^2)` auxiliary variables, so it is mainly
suited to quantum annealing or carefully scoped simulations.
