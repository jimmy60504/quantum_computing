from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator


def main() -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    backend = BasicSimulator()
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=1024)
    counts = job.result().get_counts()

    print("Running on local simulator: BasicSimulator")
    print("Circuit:")
    print(qc.draw(output="text"))
    print("Measurement counts:", counts)


if __name__ == "__main__":
    main()
