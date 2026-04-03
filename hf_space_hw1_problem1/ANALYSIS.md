# Data Reuploading Regression Analysis

This problem is less about memorizing a target formula and more about understanding what a small data reuploading circuit can infer from a difficult extrapolation split. The viewer is meant to show that story from geometry, error maps, and frequency content at the same time.

## What the circuit is actually doing

Each input pair `x1, x2` is first passed through a small classical linear layer, then squashed by `tanh`, and finally turned into rotation angles for the qubits. In every layer, those same encoded angles are re-injected through `RY` and `RZ` rotations, followed by entangling `CNOT` gates and trainable `Rot` gates. The model therefore learns a hybrid representation: classical layers adapt the interface, while the quantum circuit provides a structured nonlinear feature map.

## Why the train/test split is so hard

The train domain lives in the lower-left region, where the target surface only exposes a relatively small high-value arc of `sin(exp(x1) + x2)`. The test domain sits in the upper-right region, where the surface bends downward much more strongly. As a result, the model can fit train data very well while still struggling on test data, because this task is dominated by extrapolation rather than interpolation.

## How to read the viewer

The train 3D panel shows whether the learned surface can match the observed samples in the easy region. The test 3D panel shows how that same surface extends into unseen territory. The two error maps make the generalization gap much easier to locate spatially, while the log-scale loss plot below reveals whether later updates still improve the fit or only make tiny train-side refinements.

## What we observed about capacity

Increasing qubits or layers does increase expressive power, but the results are not monotonic. Some larger configurations drive train MSE extremely low yet still fail to reconstruct the test-domain shape in a convincing way. In other words, additional capacity helps, but it does not automatically solve the missing-information problem caused by the split itself. In our runs, a smaller model can sometimes generalize as well as, or better than, a deeper one.

## What the model does not know

The circuit never receives the symbolic rule `sin(exp(x1) + x2)`. It only sees local supervision from the train domain and tries to continue that geometry. This is why the test surface can look like the model is "trying hard" but still not truly understanding the underlying sinusoidal structure: it is learning from local shape, not from explicit knowledge of the generating function.

## Why the Fourier spectrum matters

The Fourier view explains the same behavior in frequency space. When the trained model mainly captures lower-frequency components, the resulting surface looks smooth and plausible but misses sharper oscillatory structure. This matches the reference analysis for data reuploading circuits: circuit depth affects which Fourier modes can be represented, so comparing the target and learned spectra helps explain why some models still plateau on test error even after train loss becomes very small.
