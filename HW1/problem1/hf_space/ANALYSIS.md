# Problem 1 分析

這個頁面現在不再主打舊的 qubit/layer 大矩陣，而是改成沿著題目本身的結構，從「可以精確做出答案的量子路徑」一步一步往外放鬆。

## 題目背景

Problem 1 的目標函數是 `f(x1, x2) = sin(exp(x1) + x2)`。資料切法不是一般的隨機插值，而是刻意把 train domain 放在 `[0.0, 0.5] x [0.0, 0.5]`，test domain 放在 `[0.5, 1.0] x [0.5, 1.0]`。換句話說，模型必須先在左下角看一小塊曲面，再把這個結構推到右上角沒有直接看過的區域。

![Train/test split overview](./assets/problem1_data_overview.png)

這也是為什麼這一題的重點不是單純把 train loss 壓低，而是看模型能不能保住正確的曲率與相位結構。

## 為什麼要改成結構化主線

前一版比較像是從 generic ansatz 出發，用很多 projection、encoding 和電路排列組合去猜。那條路的問題是模型雖然有很多自由度，但很難判斷到底是量子路徑不夠，還是前面的 generic 設計先把題目結構洗掉了。

後來我們發現這題其實有一個非常乾淨的量子解：

- 同一個 qubit 上連續做 `RY(exp(x1))`
- 再做 `RY(x2)`
- 最後補一個 `RY(-pi/2)`
- 量測 `PauliZ`

因為同軸 reupload 會把角度加起來，所以這條路徑幾乎就是直接在量子電路裡實現 `sin(exp(x1) + x2)`。

## 目前這個頁面裡的幾組結果在看什麼

現在的主線可以分成幾層：

1. `quantum_exact`
2. `phase_learnable`
3. `scaled_exact`
4. `same_axis_reupload`
5. `same_axis_poly`
6. `same_axis_raw`
7. `same_axis_twoqubit`
8. `twoqubit_no_reupload`
9. `twoqubit_raw_no_reupload`

前三個是從 exact 解一步一步放鬆：

- `quantum_exact`：直接用固定結構做答案
- `phase_learnable`：只把 `-pi/2` 改成可學參數
- `scaled_exact`：再讓 `exp(x1)` 和 `x2` 前面可以學 scale / bias

第四個 `same_axis_reupload` 才是第一個比較像「真正 data reuploading 模型」的版本：

- 還是保留同一個 qubit、同一個 rotation axis
- 但不再只做單一 block
- 而是用多個 reupload blocks 去學同一個 backbone

後面幾層則是在回答兩個更尖銳的問題：

- 如果把 `exp(x1)` 這個先驗拿掉，模型能不能自己長出來？
- 如果換成 2 qubits，資料組合是不是可以不用再靠 reupload？

## 這一輪最重要的結論

這幾組結果可以分成三段來看。

第一段是 exact family：

- `quantum_exact` final test MSE 約 `7.34e-15`
- `phase_learnable` final test MSE 約 `5.77e-11`
- `scaled_exact` final test MSE 約 `2.06e-10`

第二段是 1-qubit same-axis 泛化：

- `same_axis_reupload (q1, l2)` best test MSE 約 `4.91e-03`
- `same_axis_poly (q1, l2)` best test MSE 約 `2.88e-02`
- `same_axis_raw (q1, l2)` best test MSE 約 `3.22e-02`

第三段是 2-qubit 對照：

- `same_axis_twoqubit (q2, l2)` best test MSE 約 `6.95e-02`
- `twoqubit_no_reupload (q2, l1)` final test MSE 約 `5.91e-15`
- `twoqubit_raw_no_reupload (q2, l1)` best test MSE 約 `2.03e-01`

這代表三件事。

第一，題目本身不是量子模型做不到。只要保住正確的組合結構，無論是 1-qubit same-axis reupload，還是 2-qubit no-reupload exact construction，都可以非常接近答案。

第二，`same_axis_reupload` 這條 generalized data-reuploading 主線是對的。它雖然還不能 exact-fit，但學到的已經不是亂的面，而是沿著正確幾何家族往答案靠近。

第三，真正困難的地方不是「有沒有 entanglement」或「qubit 數夠不夠」，而是模型能不能自己恢復 `exp(x1)` 這個內部表示。這也是為什麼 `same_axis_poly`、`same_axis_raw`、`twoqubit_raw_no_reupload` 都比 feature-aware 的版本弱很多。

## 怎麼看 `same_axis_reupload`

如果只看數字，`same_axis_reupload-q1-l2-e10` 還有明顯誤差：

- final train MSE 約 `6.79e-05`
- final test MSE 約 `6.42e-03`

但這組最重要的不是它還差多少，而是它已經學到對的幾何家族。也就是說，現在的誤差比較像校準還沒完全對齊，而不是模型根本沒抓到題目的主結構。

這也是為什麼目前更合理的方向，是沿著 same-axis backbone 做小幅泛化，而不是回到舊的 projection-heavy 架構。

相對地，`same_axis_poly` 和 `same_axis_raw` 的結果也很有用。它們表示只要 backbone 還在，模型就不會完全走偏；但一旦把 `exp(x1)` 這個強先驗拿掉，誤差就會明顯放大。這讓我們更有把握地說：現在最主要的難題，其實是如何讓模型自己長出 `exp(x1)`，而不是如何重新發明一個更亂的 ansatz。

## 怎麼看 2-qubit 結果

2-qubit 結果很適合拿來釐清一個常見直覺。

如果已經把 `exp(x1)` 當 feature 提供給模型，那 `twoqubit_no_reupload` 其實根本不需要 repeated reupload。做法是：

- `q0` encode `exp(x1)`
- `q1` encode `x2`
- 再讀 `⟨X0 Z1⟩` 和 `⟨Z0 X1⟩`

這樣就可以直接用和角公式做出 `sin(exp(x1) + x2)`。所以 2 qubits 並不是只能回到 generic entangling ansatz；如果結構對，它甚至可以完全不靠 reupload。

但 `twoqubit_raw_no_reupload` 也說明了另一面：如果只丟 raw `x1, x2`，再補一個最小 entangling block，模型並不會自動學出 `exp(x1)`。換句話說，entanglement 不是魔法，重點還是內部表示能不能先長對。

## 這個頁面要怎麼看

最建議的看法是：

1. 先看 `quantum_exact`，把它當成結構參考解。
2. 再看 `phase_learnable` 和 `scaled_exact`，確認小幅放鬆後是不是還能維持這個解。
3. 接著看 `same_axis_reupload`，觀察真正的 data reuploading 版本離這個參考解還差多少。
4. 再看 `same_axis_poly` 和 `same_axis_raw`，把焦點放在「拿掉 `exp(x1)` 先驗之後，模型還剩多少能力」。
5. 最後對照 `twoqubit_no_reupload` 和 `twoqubit_raw_no_reupload`，理解 2-qubit 路線到底是在解「變數組合」還是在解「內部表示」。
6. 用 slider 拖過訓練過程，特別注意 test domain 曲面的曲率是怎麼長出來的。

## 頻譜在這裡的角色

Fourier spectrum 在這個頁面的用途，不是單純補一張漂亮圖，而是幫忙確認模型是不是抓到了對的主頻結構。

`quantum_exact` 的主頻和 target 幾乎重合，這和它幾乎零誤差是一致的。`same_axis_reupload` 雖然幅度還沒有完全對上，但主頻位置已經沿著正確方向靠近 target。這個訊號和 3D surface 上看到的曲率學習，是互相對應的。

## 目前的工作假設

到目前為止，最合理的假設是：

- Problem 1 最重要的 inductive bias 是同 qubit、同軸 reupload
- 一旦把 `exp(x1)` 和 `x2` 拆散，或先丟進過度 generic 的 projection，模型就比較容易偏掉
- 2-qubit 路線不是不能做，而是只有在內部表示已經對的時候，no-reupload 組合才會很乾淨
- 真正值得做的泛化，不是丟掉 exact backbone，而是保留它，再一小步一小步增加自由度
- 目前最核心的開放問題，不是「要不要更多 qubits」，而是「如何讓模型自己近似出 `exp(x1)`，同時不破壞後面和 `x2` 的乾淨組合」
