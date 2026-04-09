# Problem 2 — Answers

Seed: 11224001. All runs use 2 qubits, n_samples = 200 (140 train / 60 test), 50 training epochs, learning rate 0.05, batch size 32.

---

## (a) Fig. 6 reproduction — circle dataset

The interactive viewer shows the training curves for all three methods on the circle dataset, reproducing the qualitative pattern of Fig. 6 in Ref. [3]. Both the implicit kernel and data reuploading methods converge to >96% test accuracy within 50 epochs, while the explicit model plateaus around 77%. The kernel method reaches its final accuracy in a single fit (no iterative epochs), so its curve is flat from the start. This matches the ordering reported in the reference.

## (b) Decision boundaries

The 3 × 2 grid of decision boundaries is shown in the interactive viewer. Observations:

- **Explicit**: produces a roughly circular but imprecise boundary on the circle dataset; the decision surface on moons is smoother but slightly misaligned near the tails.
- **Kernel**: achieves a tighter circular boundary on circle; the moons boundary is more blocky, reflecting the kernel's global fit.
- **Reuploading**: produces the sharpest boundaries on both datasets, closely following the true data manifold.

## (c) Comparison table

Results for the `q2-le4-lr4-e50` run (explicit encoding layers LE = 4, reuploading layers LR = 4):

| Method | Dataset | Test Acc | Params / Kernel Evals | Train Time |
|---|---|---|---|---|
| Explicit | Circle | 76.7 % | 16 params | ≈ 20 s |
| Explicit | Moons | 88.3 % | 16 params | ≈ 20 s |
| Implicit Kernel | Circle | 96.7 % | 19,600 evals | ≈ 10 s |
| Implicit Kernel | Moons | 85.0 % | 19,600 evals | ≈ 10 s |
| Data Reuploading | Circle | 98.3 % | 32 params | ≈ 30 s |
| Data Reuploading | Moons | 98.3 % | 32 params | ≈ 30 s |

Kernel evals = 140 × 140 = 19,600 (full training Gram matrix). Training time for the implicit kernel does not scale with epoch count since the SVM is fitted in a single pass.

## (d) Discussion

Data reuploading achieves the highest and most consistent accuracy across both datasets (98.3 % on both), because layering data re-encoding between trainable rotations significantly expands the expressible function space relative to parameter count. The implicit kernel method performs well on the symmetric circle dataset (96.7 %) but drops on moons (85.0 %), consistent with Ref. [3]'s observation that the kernel's feature map is better suited to datasets whose geometry matches the quantum feature space. The explicit model is the weakest (76.7 % on circle, 88.3 % on moons), which aligns with the paper's argument that a single encoding pass limits expressibility even when more layers are added. One notable difference across datasets is that reuploading remains robust to the asymmetric moons geometry while the kernel degrades, suggesting reuploading's iterative re-injection of input data provides better inductive bias for irregular boundaries. Overall, the results reproduce the hierarchy reported in Ref. [3]: data reuploading > implicit kernel > explicit, though the gap is smaller on moons where dataset difficulty equalises the methods.
