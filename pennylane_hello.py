import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=1)
target_expval = -1.0


@qml.qnode(dev, interface="autograd")
def expval_circuit(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="autograd")
def prob_circuit(theta):
    qml.RY(theta, wires=0)
    return qml.probs(wires=0)


def loss(theta):
    return (expval_circuit(theta) - target_expval) ** 2


def main() -> None:
    theta = np.array(0.1, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    print("PennyLane hello world: train a 1-qubit circuit toward the |1> state")
    print("Target expectation <Z> =", target_expval)
    print()

    for step in range(21):
        expval = float(expval_circuit(theta))
        probs = prob_circuit(theta)
        current_loss = float(loss(theta))

        if step % 5 == 0 or step == 20:
            print(
                f"step={step:02d} "
                f"theta={float(theta): .4f} "
                f"expval={expval: .4f} "
                f"loss={current_loss: .6f} "
                f"probs=[{float(probs[0]):.4f}, {float(probs[1]):.4f}]"
            )

        if step < 20:
            theta = opt.step(loss, theta)

    print()
    print("Circuit:")
    print(qml.draw(expval_circuit)(theta))


if __name__ == "__main__":
    main()
