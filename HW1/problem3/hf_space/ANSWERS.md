# Problem 3 — Answers

Seed: 11224001. Dataset: CIFAR-10 (50,000 train / 10,000 test). CNN backbone is fixed (not modified).

---

## (a) Quantum circuit design

### Data encoding

The 256-dimensional CNN feature vector is first reduced by a classical linear layer:

```
Linear(256 → n_qubits) + tanh(·) × π
```

This compresses the feature space to one value per qubit, then scales the result to the encoding range [−π, π]. Each value is then encoded onto its corresponding qubit via an RY rotation:

```
RY(x_i) on qubit i,  for i = 0, …, n_qubits − 1
```

### Variational circuit

Each of the L variational layers consists of single-qubit trainable rotations followed by a CNOT entangling ring:

```
for each layer ℓ = 1, …, L:
    RY(θ_{ℓ,i}) · RZ(φ_{ℓ,i})  for each qubit i
    CNOT(i → (i+1) mod n_qubits)  for each qubit i     [ring topology]
```

Total quantum parameters: L × n_qubits × 2.

### Measurement and post-processing

The PauliZ expectation value ⟨Z_i⟩ is measured on each qubit, yielding an n_qubits-dimensional real vector. A final classical linear layer maps these expectations to 10-class logits:

```
Linear(n_qubits → 10) → class logits
```

### Configuration used

| Hyperparameter | Value |
|---|---|
| n_qubits | 8 |
| Variational layers L | 4 |
| Diff. method | adjoint (lightning.gpu) |
| Backbone | Frozen |

Trainable parameters: Linear(256→8) [2,056] + weights [64] + Linear(8→10) [90] = **2,210 total**.

---

## (b) Test accuracy

| Model | Test Accuracy |
|---|---|
| CNN + MLP (baseline) | 33.1 % (1 epoch) — full training in progress |
| CNN + QNN (hybrid) | 29.6 % (1 epoch) — full training in progress |

*(Results above are from a 1-epoch pilot run with frozen backbone. Full 20-epoch results will be updated here.)*

---

## (c) Comparison table

| Model | Test Acc | Trainable Params | Training Time |
|---|---|---|---|
| CNN + MLP | 33.1 % (1 ep.) | 2,570 | 30.8 s / epoch |
| CNN + QNN | 29.6 % (1 ep.) | 2,210 | ≈ 460 s / epoch (lightning.gpu) |

The MLP head is a single `Linear(256, 10)` layer. The QNN head adds a classical pre-layer, 64 quantum parameters, and a classical post-layer, for a slightly smaller trainable parameter count than the MLP.

---

## (d) Training curves

Training loss and test accuracy curves for both models (plotted per epoch) are shown in the interactive viewer after full training completes.

---

## (e) Discussion

The hybrid CNN + QNN model uses a parameterized quantum circuit as a drop-in replacement for the MLP classification head, encoding CNN features into qubit rotations and extracting class information from Pauli-Z expectation values. With a frozen backbone and 8 qubits, the QNN operates in a highly compressed latent space (8-dimensional), which limits its expressibility relative to the 256-dimensional MLP head but drastically reduces trainable parameters. The key design choice is the classical pre- and post-processing layers: the pre-layer learns which linear projection of the 256-D feature vector is most informative to encode into the circuit, while the post-layer maps quantum readouts to class logits. The CNOT ring entanglement structure introduces qubit correlations at each layer, enabling the circuit to capture feature interactions that pure angle encoding without entanglement could not represent. Whether the QNN can match or exceed the MLP baseline on CIFAR-10 is an open question that depends on whether the quantum feature map offers any advantage over a linear classifier in the frozen-backbone setting; initial results suggest a small performance gap, which is consistent with findings in the literature for near-term quantum classifiers on classical benchmark datasets.
