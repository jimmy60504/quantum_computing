# Problem 1 Modeling Notes

## 2026-04-05: Loosened input compression is not the main path

Tested setting:

- model family: `exp`, `q=2`, `l=2`
- change: remove `tanh` by using `input_activation=identity`
- change: enlarge projected angles with `angle_scale=pi`

Run names:

- `exp-q2-l2-actidentity-aspi-e20`
- `exp-q2-l2-actidentity-aspi-e50`

Observed behavior:

- This setting drives training loss down very aggressively.
- Test performance improves from `e20` to `e50`, so the run is not fully saturated at 20 epochs.
- However, the overall direction still looks weak for generalization and should not be treated as the main path forward.

Key metrics:

- `e20` final train MSE: `1.541e-4`
- `e20` final test MSE: `3.392e-1`
- `e50` final train MSE: `4.514e-6`
- `e50` final test MSE: `2.173e-1`
- best test MSE stayed at step `5`: `1.106e-1`

Interpretation:

- Longer training partially repairs the badly degraded test surface.
- The best test point does not move later in training, even after 50 epochs.
- This suggests the setting is more like "overshoot early, then slowly recover" than "learn the correct downward bend."
- For Problem 1, this is useful as a negative result, but it does not look like the right primary modeling direction.

Practical takeaway:

- Do not prioritize `identity + pi` as the next main branch of experiments.
- Retire this branch from active code and keep it only as recorded history.

## 2026-04-05: Cosine scheduler did not rescue the baseline path

Tested setting:

- model family: `exp`, `q=2`, `l=2`
- baseline input path kept unchanged: `input_activation=tanh`, `angle_scale=1`
- optimizer change: cosine decay from `lr=1e-3` down to `min_lr=1e-5`
- training length: `50` epochs

Run name:

- `exp-q2-l2-lr1e-3-cosine-min1e-5-e50-trainonly`

Observed behavior:

- The scheduler gives a smoother training schedule than the fixed large-learning-rate runs.
- However, it still does not recover a good test surface.
- This suggests learning-rate shape is not the main bottleneck for the current baseline architecture.

Key metrics:

- final train MSE: `1.096e-3`
- final test MSE: `5.990e-1`
- best test MSE: `5.291e-1`

Interpretation:

- Reducing the step size over time does not by itself fix the extrapolation failure.
- The current failure mode is likely more structural than purely optimization-speed-related.

Practical takeaway:

- Record scheduler sweeps as a useful negative result.
- Return the main line of investigation to the original baseline configuration rather than continuing to tune scheduler shape first.
- Retire this branch from active code and keep it only as recorded history.

## 2026-04-05: Two answer-aware sanity checks

### 1. Classical oracle shortcut

Tested setting:

- mode: `oracle_shortcut`
- idea: inject `sin(exp(x1) + x2)` directly as an answer-aware classical shortcut to the output
- run name: `oracle-shortcut-q2-l2-e5`

Observed behavior:

- The model fits immediately.
- Batch loss is effectively zero from the beginning.

Key metrics:

- best test MSE: `0.0`
- final train MSE: `0.0`
- final test MSE: `0.0`

Interpretation:

- The data, target definition, training loop, snapshot export, metrics pass, and viewer pipeline are all behaving correctly.
- If the exact answer is injected directly into the output path, the system has no trouble reproducing the surface.
- This confirms the main failure in baseline runs is not a bookkeeping or evaluation bug.

Status:

- This shortcut was useful as a diagnostic, but it is intentionally not part of the cleaned active code path.

### 2. Quantum exact construction

Tested setting:

- mode: `quantum_exact`
- idea: use a 1-qubit hand-crafted reuploading circuit that mirrors the target structure
- run name: `quantum-exact-q1-l1-e5`

Circuit idea:

- upload `exp(x1)` with `RY`
- upload `x2` with another `RY`
- apply a fixed `RY(-pi/2)`
- measure `PauliZ`

Why this works:

- same-axis reuploading adds the angles
- `RY(exp(x1))` then `RY(x2)` gives an effective angle of `exp(x1) + x2`
- shifting by `-pi/2` turns the final `PauliZ` expectation from cosine into sine
- this realizes `sin(exp(x1) + x2)` up to numerical precision

Key metrics:

- best test MSE: `1.340e-07`
- final train MSE: `5.215e-08`
- final test MSE: `4.978e-07`

Interpretation:

- A quantum path can represent the target essentially perfectly.
- The issue is therefore not "quantum data reuploading cannot express this function."
- The issue is that our original baseline architecture is too generic and destroys the problem-aligned structure before measurement.

Practical takeaway:

- Keep both answer-aware checks as reference diagnostics.
- Use `quantum_exact` as the constructive proof that a structured reuploading design can solve the task.
- Future modeling should move from the generic learned projection path toward architectures that preserve same-axis additive phase structure.

## Active direction

The main experiment path should now follow a structured generalization ladder instead of the earlier generic baseline:

- `quantum_exact`
- `phase_learnable`
- `scaled_exact`
- `same_axis_reupload`

Rationale:

- Start from the exact same-axis reuploading construction that is known to solve the task.
- Relax one assumption at a time.
- Avoid returning to the old projection-heavy baseline as the main search space, because that architecture was introduced as a generic guess rather than something implied by the problem structure.

## 2026-04-06: Structured exact family converges cleanly, and a first generalized reuploading model preserves the right shape

### 1. Exact-family sanity sweep at `e10`

Tested settings:

- `quantum_exact`, `q=1`, `l=1`
- `phase_learnable`, `q=1`, `l=1`
- `scaled_exact`, `q=1`, `l=1`

Run names:

- `quantum-exact-q1-l1-e10`
- `phase-learnable-q1-l1-e10`
- `scaled-exact-q1-l1-e10`

Key metrics:

- `quantum_exact` final test MSE: `7.345e-15`
- `phase_learnable` final test MSE: `5.770e-11`
- `scaled_exact` final test MSE: `2.055e-10`

Interpretation:

- The exact construction remains a clean proof that the target is solved by same-axis angle addition.
- Letting the phase become learnable still converges back to the intended solution.
- Letting both phase and feature scale/bias become learnable also converges, with only a modest loss in precision.

Practical takeaway:

- The structured ladder is behaving as intended.
- Small relaxations of the exact circuit do not immediately destroy the solution.

### 2. First generalized data-reuploading attempt

Tested setting:

- `same_axis_reupload`, `q=1`, `l=2`
- circuit pattern per block: same-qubit `RY(exp(x1)) -> RY(x2) -> RY(phi)`
- each block learns its own feature scale, feature bias, and phase shift

Run name:

- `same-axis-reupload-q1-l2-e10`

Key metrics:

- best test MSE: `4.910e-03`
- final test MSE: `6.424e-03`
- final train MSE: `6.792e-05`

Observed behavior:

- This model does not exact-fit the task the way the hand-crafted ladder does.
- However, it clearly learns the right surface family rather than collapsing to a flat or unrelated shape.
- The learned surface tracks the target curvature much better than the retired generic baseline.

Interpretation:

- Preserving same-qubit, same-axis additive structure appears to be the right inductive bias for this problem.
- The first generalized reuploading model is already much closer to a meaningful data-reuploading solution than the old projection-heavy architecture.
- The remaining error looks more like calibration or readout mismatch than a total failure to represent the correct geometry.

Practical takeaway:

- Keep `same_axis_reupload` as the main generalized branch.
- Future relaxations should stay attached to this backbone rather than returning to the older generic ansatz family.

## 2026-04-06: Relaxing the exact backbone shows where the hard part really is

After confirming that the structured exact family works, the next question was how to relax the model toward a more general data-reuploading design without losing the problem-aligned backbone.

### 1. Learning `exp(x1)` is harder than preserving the same-axis geometry

Tested settings:

- `same_axis_poly`, `q=1`, `l=2`
- `same_axis_raw`, `q=1`, `l=2`

Run names:

- `same-axis-poly-q1-l2-e10`
- `same-axis-raw-q1-l2-e10`

Key metrics:

- `same_axis_poly` best test MSE: `2.884e-02`
- `same_axis_poly` final test MSE: `1.577e-01`
- `same_axis_raw` best test MSE: `3.217e-02`
- `same_axis_raw` final test MSE: `1.296e-01`

Interpretation:

- Both models keep the same-qubit, same-axis additive backbone, so they still move in a visibly meaningful direction rather than collapsing to an unrelated surface.
- However, once the hand-crafted `exp(x1)` feature is removed or replaced by a learned low-order proxy, performance drops substantially relative to `same_axis_reupload`.
- This suggests the main challenge is no longer "can the circuit express the right type of geometry?" but rather "can the model discover the right nonlinear transformation of `x1` on its own?"

Practical takeaway:

- The same-axis backbone remains the right structural prior.
- Learning the inner `exp(x1)` transformation from weaker priors is possible in spirit, but clearly harder and less stable than giving the model the right feature directly.
- Future relaxation steps should treat "recovering `exp(x1)`" as the main difficulty, not same-axis composition itself.

### 2. A second qubit does not automatically help if it breaks the main structural story

Tested setting:

- `same_axis_twoqubit`, `q=2`, `l=2`

Run name:

- `same-axis-twoqubit-q2-l2-e10`

Key metrics:

- best test MSE: `6.953e-02`
- final test MSE: `1.312e-01`
- final train MSE: `4.125e-04`

Interpretation:

- This design keeps a same-axis main path on `q0` and uses `q1` as a residual path.
- The extra qubit lowers train error, but it does not improve generalization over the simpler 1-qubit raw same-axis model.
- In other words, extra capacity alone is not the missing ingredient here.

Practical takeaway:

- Do not assume "more qubits" is automatically the next step toward a better Problem 1 model.
- A second qubit is only useful if it preserves or sharpens the right compositional structure.

### 3. Two-qubit exact composition works without reupload if the right feature is provided

Tested setting:

- `twoqubit_no_reupload`, `q=2`, `l=1`

Run name:

- `twoqubit-no-reupload-q2-l1-e1`

Construction idea:

- encode `exp(x1)` on `q0`
- encode `x2` on `q1`
- do not use data reuploading
- read out `⟨X0 Z1⟩` and `⟨Z0 X1⟩`
- sum the two observables to realize the sine angle-addition identity

Key metrics:

- best test MSE: `5.907e-15`
- final train MSE: `1.055e-15`
- final test MSE: `5.907e-15`

Interpretation:

- Problem 1 does not require data reuploading in an absolute sense.
- A 2-qubit circuit can solve the task exactly in one shot if it is given the right intermediate feature, namely `exp(x1)`.
- The hard part is therefore not "1 qubit versus 2 qubits" or "reupload versus no reupload" by itself.
- The hard part is whether the model has access to the right inner representation before composing it with `x2`.

Practical takeaway:

- Treat this result as an exact constructive counterexample to the idea that repeated reuploading is the only way to solve the task quantumly.
- Keep the focus on how the architecture represents `exp(x1)`, not only on qubit count.

### 4. Entanglement plus raw inputs is still not enough in the minimal 2-qubit no-reupload setting

Tested setting:

- `twoqubit_raw_no_reupload`, `q=2`, `l=1`

Run name:

- `twoqubit-raw-no-reupload-q2-l1-e10`

Construction idea:

- encode raw `x1` and `x2` once
- add one entangling `CNOT`
- follow with a small trainable rotation block and mixed two-body readout

Key metrics:

- best test MSE: `2.028e-01`
- final test MSE: `2.908e-01`
- final train MSE: `3.630e-03`

Interpretation:

- This model does learn something, so the entangling path is not completely degenerate.
- However, it is clearly worse than the 1-qubit same-axis raw model and much worse than the feature-aware exact 2-qubit construction.
- A small entangling block is therefore not enough to make the model infer the missing `exp(x1)` structure from raw inputs in one shot.

Practical takeaway:

- "Just add entanglement" is not a sufficient answer for Problem 1.
- If raw inputs are used without reuploading, the main missing capability is still the discovery of the nonlinear inner feature, not merely variable interaction.

## Updated working picture

The current modeling picture is:

- `same_axis_reupload` shows that preserving same-axis additive composition is the right generalized 1-qubit backbone.
- `same_axis_poly` and `same_axis_raw` show that relaxing the explicit `exp(x1)` prior is possible, but this is now the dominant source of error.
- `same_axis_twoqubit` shows that extra capacity does not help by itself.
- `twoqubit_no_reupload` shows that 2 qubits can solve the task exactly without reuploading if the right feature is already available.
- `twoqubit_raw_no_reupload` shows that a minimal entangling ansatz on raw inputs does not yet recover that feature.

The most important conclusion is that the central modeling problem has become:

- how to let the model recover or approximate `exp(x1)` while preserving a composition rule that still combines cleanly with `x2`

rather than:

- whether the circuit needs more generic freedom in an unconstrained sense.
