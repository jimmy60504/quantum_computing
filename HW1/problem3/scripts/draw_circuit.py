"""Draw the Problem 3 quantum circuit using PennyLane's built-in drawer.

Generates a clean circuit diagram showing:
  - Angle encoding: RY(xᵢ) on each qubit
  - 4 variational layers: RY(θ) + RZ(φ) + CNOT ring

Saves to HW1/problem3/report_figs/prob3_circuit.png
        HW1/problem3/report_figs/prob3_circuit.pdf
"""

from pathlib import Path
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_QUBITS = 8
NUM_LAYERS = 4

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def circuit(inputs, weights):
    # ── Encoding ──────────────────────────────────────────────
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)

    # ── Variational layers ────────────────────────────────────
    for layer in range(NUM_LAYERS):
        for i in range(NUM_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(NUM_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % NUM_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# Dummy inputs for drawing
inputs  = np.zeros(NUM_QUBITS)
weights = np.zeros((NUM_LAYERS, NUM_QUBITS, 2))

fig, ax = qml.draw_mpl(circuit, decimals=None, style="pennylane")(inputs, weights)

# Tweak figure size for readability
fig.set_size_inches(20, 7)
ax.set_title(
    f"Problem 3 Quantum Circuit  ({NUM_QUBITS} qubits × {NUM_LAYERS} layers)\n"
    "Encoding: RY(xᵢ)   |   Variational: [RY(θ) · RZ(φ)] × L + ring CNOT",
    fontsize=11,
    pad=10,
)
fig.tight_layout()

OUT = Path(__file__).parents[1] / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

for ext in ["png", "pdf"]:
    path = OUT / f"prob3_circuit.{ext}"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"saved {path}")

plt.close(fig)
