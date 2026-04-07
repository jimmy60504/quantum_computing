# Problem 1 作答

## (a) Best model 的 training/test loss curve

以 data reuploading 架構為主線，`same_axis_reupload (q=1, l=2)` 為最佳配置。其 final test MSE ≈ 6.42e-3，低於題目要求的 0.1，且 training 與 test loss 均單調下降，無過擬合現象。

作為對比，`quantum_exact`、`phase_learnable`、`scaled_exact` 的 loss curve 幾乎在前幾個 epoch 就收斂至接近零——這說明此題的關鍵在於電路是否保住了正確的 same-axis additive structure，而非模型本身的訓練能力。

## (b) Hyperparameter/configuration 比較表

以下四組涵蓋主要的結構差異：

| 模型 | Q | L | `exp(x1)` 先驗 | Best Test MSE |
|---|---|---|---|---|
| `same_axis_reupload` | 1 | 2 | ✓ | 4.91e-3 |
| `same_axis_poly` | 1 | 2 | 近似 | 2.88e-2 |
| `same_axis_raw` | 1 | 2 | ✗ | 3.22e-2 |
| `same_axis_twoqubit` | 2 | 2 | ✓ | 6.95e-2 |

補充診斷性對比：

| 模型 | Q | L | `exp(x1)` 先驗 | Best Test MSE |
|---|---|---|---|---|
| `twoqubit_no_reupload` | 2 | 1 | ✓ | 5.91e-15 |
| `twoqubit_raw_no_reupload` | 2 | 1 | ✗ | 2.03e-1 |

主要觀察：移除 `exp(x1)` 先驗後誤差上升約一個數量級；增加至 2 qubits 並不保證改善，`same_axis_twoqubit` 反比 1-qubit 版本差，因為糾纏引入的參數空間更大而收斂更困難。

## (c) Fourier spectrum 討論

Fourier spectrum 將 3D surface 壓縮到 2D 頻域，用於判斷模型學到的曲面是否與 target 屬於同一個函數族：主頻位置對應整體幾何結構，幅度對應各頻率成分的強度。

- **`quantum_exact`**：spectrum 與 target 幾乎完全重合，與接近零的 MSE 一致。
- **`same_axis_reupload`**：主頻位置已對齊 target，但幅度尚未完全校準，對應 3D surface 上「曲率方向正確但有殘差」的現象。
- **`twoqubit_raw_no_reupload`**：只捕捉到部分主頻，與其 3D surface 僅學到局部趨勢的結果一致。

相比單看 loss 數字，spectrum 提供了更直觀的全局診斷：可以直接區分「模型只是局部湊形狀」與「模型真正學到 target 的主頻結構」這兩種情況。
