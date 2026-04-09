# HW1 Problem 2 — QML Classifiers

Binary classification benchmark comparing three quantum machine learning approaches on two datasets (circle, moons).

| Method | Description |
|--------|-------------|
| Explicit quantum model | Encoding circuit S(x) + trainable W(θ) + measurement |
| Implicit quantum kernel | Fixed encoding kernel k(xi, xj) passed to SVM |
| Data reuploading | Interleaved encoding and trainable layers |

**[Live viewer →](https://huggingface.co/spaces/jimmy60504/QML-Classifier-Explorer)**

## gx10 setup

See [HW1/problem1/README.md](../problem1/README.md) for sync and Docker setup.

## Training

```bash
./scripts/gx10_run_py.sh HW1/problem2/train.py
```

Preview datasets locally:

```bash
conda activate quantum-computing
python -m HW1.problem2.scaffold --preview-datasets --write-plan
```

## Upload to Hugging Face

```bash
./HW1/problem2/scripts/gx10_prepare_hf_space.sh
./HW1/problem2/scripts/gx10_upload_hf_space.sh jimmy60504/QML-Classifier-Explorer
```
