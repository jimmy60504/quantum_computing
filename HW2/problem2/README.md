# HW2 Problem 2 — Max-Cut with QAOA

Max-Cut on a student-specific random 8-node graph. Implement QAOA via PennyLane.

```python
import networkx as nx
seed = 0  # replace with student ID
G = nx.gnp_random_graph(n=8, p=0.5, seed=seed)
```

Cost function: C(z) = ½ ∑_{(i,j)∈E} (1 − z_i z_j)  
Cost Hamiltonian: H_C = −½ ∑_{(i,j)∈E} Z_i Z_j + |E|/2 · I

## Sub-tasks

**(a)** Visualize the graph. Brute-force all 2^8 = 256 partitions; report edges, max-cut value, and optimal partition(s).

**(b)** QAOA depth p = 1: plot 2D energy landscape F(γ, β) = ⟨γ,β|H_C|γ,β⟩ as a heatmap over γ ∈ [0, 2π], β ∈ [0, π]. Identify the global minimum. Does the optimizer reliably find it?

**(c)** Implement QAOA for depths p ∈ {1, 2, 3, 4}. Report: best cut found, approximation ratio (found / optimal), optimized parameters.

**(d)** Also solve with `SimulatedAnnealingSampler`. Comparison table (brute-force, SA, QAOA p=1..4): best cut, approximation ratio, computation time. Brief discussion (3–5 sentences).

## References

- [PennyLane QAOA tutorial](https://pennylane.ai/qml/demos/tutorial_qaoa_intro)
