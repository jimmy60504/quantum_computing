import os
from time import perf_counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def build_demo_circuit(num_qubits: int = 20, layers: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for qubit in range(num_qubits):
        qc.h(qubit)

    for layer in range(layers):
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
        for qubit in range(num_qubits):
            qc.ry(0.05 * (layer + 1) * (qubit + 1), qubit)

    qc.save_probabilities_dict()
    return qc


def run_demo(device: str, circuit: QuantumCircuit, seed: int) -> tuple[float, dict[str, float], dict]:
    backend = AerSimulator(method="statevector", device=device)
    compiled = transpile(circuit, backend, optimization_level=0, seed_transpiler=seed)

    started = perf_counter()
    result = backend.run(compiled, seed_simulator=seed).result()
    elapsed = perf_counter() - started

    probabilities = result.data(0)["probabilities"]
    top_probabilities = dict(
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
    )
    metadata = result.results[0].metadata
    return elapsed, top_probabilities, metadata


def max_probability_delta(cpu_probabilities: dict[str, float], gpu_probabilities: dict[str, float]) -> float:
    keys = set(cpu_probabilities) | set(gpu_probabilities)
    return max(
        abs(cpu_probabilities.get(key, 0.0) - gpu_probabilities.get(key, 0.0))
        for key in keys
    )


def main() -> None:
    num_qubits = int(os.getenv("AER_DEMO_QUBITS", "20"))
    layers = int(os.getenv("AER_DEMO_LAYERS", "3"))
    seed = int(os.getenv("AER_DEMO_SEED", "7"))
    circuit = build_demo_circuit(num_qubits=num_qubits, layers=layers)

    print("Qiskit Aer GPU demo on gx10")
    print("Circuit qubits:", circuit.num_qubits)
    print("Circuit depth:", circuit.depth())
    print("Seed:", seed)
    print()

    cpu_time, cpu_probabilities, cpu_metadata = run_demo("CPU", circuit, seed)
    gpu_time, gpu_probabilities, gpu_metadata = run_demo("GPU", circuit, seed)
    delta = max_probability_delta(cpu_probabilities, gpu_probabilities)

    print(f"CPU run time: {cpu_time:.3f}s")
    print(f"GPU run time: {gpu_time:.3f}s")
    print(f"Max probability delta: {delta:.3e}")
    print()
    print("Top CPU probabilities:", cpu_probabilities)
    print("Top GPU probabilities:", gpu_probabilities)
    print()
    print("CPU metadata:", {key: cpu_metadata[key] for key in ("method", "device") if key in cpu_metadata})
    print("GPU metadata:", {key: gpu_metadata[key] for key in ("method", "device") if key in gpu_metadata})


if __name__ == "__main__":
    main()
