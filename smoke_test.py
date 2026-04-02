import importlib


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    import torch

    print("qiskit:", version_of("qiskit"))
    print("qiskit-machine-learning:", version_of("qiskit_machine_learning"))
    print("pennylane:", version_of("pennylane"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
