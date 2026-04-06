# Problem 1 回答補充

這一頁不是展示主文，而是把 Problem 1 要求的三個重點，整理成比較接近報告作答的說法。

## (a) Best model 的 training/test loss curve

如果只把「真正 data reuploading 模型」算進來，而不把 hand-crafted exact diagnostics 當成正式最佳解，那目前最值得當作 best configuration 討論的是 `same_axis_reupload (q=1, l=2)`。

理由不是只有它的數字最好，而是它同時滿足兩件事：

- 它保留 data reuploading 的核心結構
- 它的 test surface 已經沿著正確幾何家族接近 target

在這個設定下，training loss 和 test loss 都是往下降的，而且 test curve 沒有像早期 generic baseline 那樣一開始碰巧很低、後面又崩掉。它最後的 `final test MSE` 約 `6.42e-03`，已經明顯低於題目要求的 `0.1`。

如果把 `quantum_exact`、`phase_learnable`、`scaled_exact` 也放進來看，它們可以當成結構參考解：這幾組的 loss curve 幾乎是快速收斂到接近零，說明這題的關鍵不是量子模型做不到，而是模型是否保住了正確的 same-axis additive structure。

## (b) 至少 4 組 hyperparameter/configuration 的比較表

這一題原本要求比較不同 qubit 數、layer 數或 encoding 設計。對這份實驗來說，比較有意義的不是單純做大矩陣，而是比較幾個有代表性的結構化配置。最適合放進表裡的至少四組是：

- `same_axis_reupload (q=1, l=2)`
- `same_axis_poly (q=1, l=2)`
- `same_axis_raw (q=1, l=2)`
- `same_axis_twoqubit (q=2, l=2)`

如果想把診斷性結果也一起呈現，還可以補：

- `twoqubit_no_reupload (q=2, l=1)`：說明 2 qubits 在 feature 已經對的情況下可以 exact solve
- `twoqubit_raw_no_reupload (q=2, l=1)`：說明只加 entanglement 並不能自動學出 `exp(x1)`

這個比較表最重要的結論不是單純哪個 train MSE 最低，而是：

- `same_axis_reupload` 是目前最好的 data reuploading 主線
- 把 `exp(x1)` 拿掉後，`same_axis_poly` 和 `same_axis_raw` 都明顯變差
- 增加到 2 qubits 並不保證更好，關鍵還是內部表示與組合方式

也就是說，這個表真正展示的是 inductive bias 的差異，而不只是模型大小的差異。

## (c) Fourier spectrum 的討論

這一題要求頻譜分析，重點不是把它當成另一張漂亮圖，而是把它當成一個 2D 的全局表示，幫助我們判斷模型學到的曲面是否真的和 target 屬於同一個 family。

3D surface plot 很直觀，能直接看出曲面是往哪裡彎、和 target 是否大致平行；但如果只看 3D 圖，有時很難精準比較「整體結構到底差在哪」。Fourier spectrum 則把這件事壓到 2D frequency domain：

- 如果主頻位置和 target 接近，代表模型抓到了對的整體幾何結構
- 如果主頻缺失或相對強度錯很多，通常就代表模型只是在局部湊出形狀，而不是真的學到正確 family

在這組結果裡，`quantum_exact` 的 spectrum 幾乎和 target 重合，這和它近乎零誤差是一致的。`same_axis_reupload` 則已經抓到正確的主頻位置，只是幅度還沒有完全對上；這也正好對應到 3D surface 上看到的現象，也就是曲率方向已經對了，但還沒有完全校準到 target。相對地，`twoqubit_raw_no_reupload` 的 spectrum 只抓到部分結構，這也和它在 3D 圖上只學到局部趨勢、但沒有真正貼上 target 的結果一致。

因此，這一題的頻譜分析可以總結成一句話：它提供了一個比單看 loss 更容易比較、又比只看 3D surface 更全局的 2D 診斷工具，用來判斷模型是否真的學到 target surface 的主要結構。
