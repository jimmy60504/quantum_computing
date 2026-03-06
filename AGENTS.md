# AGENTS.md

## Project summary

- Repository name: `quantum_computing`
- Primary purpose: local experiments and development with Qiskit
- Preferred environment manager: Conda

## Environment assumptions

- Use the Conda environment named `quantum-computing`
- Target Python version is `3.11`
- Install dependencies from `environment.yml`
- Main runtime dependency is `qiskit==2.3.0`

## Working rules for future agents

- Prefer `conda run -n quantum-computing <command>` when running Python tooling
  non-interactively.
- Keep `README.md` and `environment.yml` in sync when dependencies change.
- Do not introduce Docker unless the user explicitly asks for containerization.
- Keep the repository lightweight and suitable for local experimentation unless
  the user requests a fuller project scaffold.
