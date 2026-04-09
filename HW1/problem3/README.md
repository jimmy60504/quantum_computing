# HW1 Problem 3 — Hybrid QNN vs MLP on CIFAR-10

Compares a classical MLP head against a quantum circuit head on top of a shared frozen CNN backbone. Both heads receive the same 256-dim CNN features and output 10-class logits.

| Model | Head | Params |
|-------|------|--------|
| CNN + MLP | Linear(256→10) | 2,570 |
| CNN + QNN | Linear(256→8)+tanh·π → RY encoding → 4×(RY+RZ+CNOT ring) → ⟨Z_i⟩ → Linear(8→10) | 2,210 |

Config: 8 qubits, 4 layers, `lightning.gpu` + adjoint diff, frozen backbone, batch size 64.

**[Live viewer →](https://huggingface.co/spaces/jimmy60504/Hybrid-QNN-Explorer)**

## gx10 setup

See [HW1/problem1/README.md](../problem1/README.md) for sync and Docker setup.

## Training

Full pipeline (MLP + QNN, 20 epochs each):

```bash
# recommended: run inside tmux
tmux new -s prob3
bash HW1/problem3/scripts/gx10_hw1_prob3_pipeline.sh
# Ctrl-b d to detach; tmux attach -t prob3 to reattach
```

Environment overrides (all optional):

```bash
PROB3_Q_DEVICE=lightning.gpu \   # default: lightning.gpu
PROB3_Q_DIFF_METHOD=adjoint \    # default: adjoint
PROB3_FREEZE_BACKBONE=1 \        # default: 1 (frozen)
bash HW1/problem3/scripts/gx10_hw1_prob3_pipeline.sh
```

Single model training:

```bash
./scripts/gx10_run_py.sh HW1/problem3/train.py \
  --method qnn \
  --num-qubits 8 --num-layers 4 \
  --epochs 20 \
  --q-device lightning.gpu \
  --q-diff-method adjoint \
  --freeze-backbone
```

## Upload to Hugging Face

```bash
./HW1/problem3/scripts/gx10_prepare_hf_space.sh
./HW1/problem3/scripts/gx10_upload_hf_space.sh jimmy60504/Hybrid-QNN-Explorer
```
