from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.providers.basic_provider import BasicSimulator


def build_qft_demo(num_qubits: int = 24) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)
    qc.append(QFT(num_qubits=num_qubits), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main() -> None:
    num_qubits = 24
    qc = build_qft_demo(num_qubits)

    backend = BasicSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=1024).result()
    counts = result.get_counts()

    print(f"Running QFT demo locally with {num_qubits} qubits on BasicSimulator")
    print("Circuit depth:", qc.depth())
    print("Compiled depth:", compiled.depth())
    print("Operation counts:", qc.count_ops())
    print("Top measurement counts:", dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]))


if __name__ == "__main__":
    main()
