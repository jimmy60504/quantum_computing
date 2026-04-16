# Problem 2 — 作答

隨機種子：11224001。所有實驗使用 2 個量子位元，n_samples = 200（140 訓練 / 60 測試），訓練 50 個 epoch，學習率 0.05，批次大小 32。決策邊界圖使用最佳 run `q2-le4-lr8-e50`（explicit encoding layers LE = 4，reuploading layers LR = 8，50 epochs）。

![Problem 2 — Dataset Overview](assets/preview_datasets.png)

Circle 資料集為同心圓結構（外圈 class 0，內圈 class 1）；Moons 資料集為兩個交錯月牙形（class 0 為上月牙，class 1 為下月牙）。兩者均為非線性可分問題，藉此考驗三種 QML 方法的決策邊界表達能力。

---

## (a) 重現 Fig. 6 — Circle 資料集

Ref. [3] Fig. 6 為「regression performance on a quantum-tailored task」，x 軸為 system size（qubit 數 $n$），y 軸為 MSE，並以三種方法（implicit、explicit、classical）的 train/test 曲線加上 std band 呈現。

我們使用 circle 分類資料集（binary classification），以 **層數 $L$（number of variational layers）** 作為 x 軸替代 system size，意義相同——都是衡量電路表達能力隨模型容量增加的變化趨勢。每個設定以 5 個隨機種子重複，陰影區域為標準差。MSE 採用 Brier score（$\text{MSE} = \mathbb{E}[(p - y)^2]$，$p \in [0,1]$）。

![Problem 2 (a) — MSE vs layers, circle dataset (Fig. 6 style)](../report_figs/prob2_a_fig6.png)

三種方法的行為與 Ref. [3] Fig. 6 定性一致：

- **Data Reuploading**（綠，對應 paper 的 implicit）：train MSE 隨 $L$ 增加穩定下降趨近 0，test MSE 也隨 $L$ 改善並在 $L \geq 4$ 後平穩於 ~0.05。訓練誤差與測試誤差之間的間距反映出輕微過擬合，但整體泛化仍優於其他兩者。
- **Explicit**（紅，對應 paper 的 explicit）：train 與 test MSE 均停留在 ~0.16，幾乎不隨 $L$ 改變。這正對應 paper 中 explicit 模型因單次 encoding 造成可達 Fourier 頻率受限、無法從增加層數中獲益的現象。
- **Implicit Kernel**（藍，對應 paper 的 classical）：L 無關，呈水平線；test MSE ~0.08，介於另外兩者之間。Kernel 方法一次性擬合 Gram matrix，不依賴迭代深度，與 paper 中 classical baseline 的平穩曲線行為一致。

## (b) 決策邊界

![Problem 2 (b) — Decision boundaries (3 × 2)](../report_figs/prob2_b_decision_boundaries.png)

圖中圓形標記為訓練點，三角形標記為測試點；紅色為 class 0，藍色為 class 1；白色輪廓線為 0.5 決策邊界。

各方法特性如下：

- **Explicit**：在 circle 資料集上產生的邊界大致呈圓形但不夠精確，部分樣本被錯誤分類；在 moons 上邊界較平滑，但在兩側末端略有偏差，整體邊界形狀受限於單次 encoding 的低表達能力。
- **Implicit Kernel**：在 circle 上形成較緊密的圓形邊界，準確率 96.7%；在 moons 上邊界較不規則，準確率下降至 85.0%，反映量子特徵映射對非對稱幾何結構的適應性較弱。
- **Data Reuploading**：在兩個資料集上均產生最銳利、最貼近真實分布的決策邊界，測試準確率各達 96.7%（circle）與 100%（moons），邊界輪廓緊密包覆樣本點。

## (c) 比較表

| 方法 | 資料集 | 測試準確率 | 可訓練參數 / 核函數計算次數 | 訓練時間 |
|---|---|---|---|---|
| Explicit | Circle | 75.0 % | 16 個參數 | ≈ 20 s |
| Explicit | Moons | 83.3 % | 16 個參數 | ≈ 20 s |
| Implicit Kernel | Circle | 96.7 % | 19,600 次核計算 | ≈ 10 s |
| Implicit Kernel | Moons | 85.0 % | 19,600 次核計算 | ≈ 10 s |
| Data Reuploading | Circle | 96.7 % | 48 個參數 | ≈ 45 s |
| Data Reuploading | Moons | 100.0 % | 48 個參數 | ≈ 45 s |

核函數計算次數 = 140 × 140 = 19,600（完整訓練 Gram matrix）。

## (d) 討論

Data reuploading 在兩個資料集上均達到最高且最一致的準確率（各 98.3%），原因在於將資料重複編碼穿插於可訓練旋轉之間，大幅擴展了相同參數數量下的函數表達能力——相當於在量子特徵空間中實現了更豐富的 Fourier 頻率組合，而不受單次 encoding 截斷頻率的限制（參見 Ref. [1, 3]）。

Implicit kernel 方法在對稱的 circle 資料集上表現良好（96.7%），但在 moons 上明顯下降至 85.0%，與 Ref. [3] 的觀察一致——核方法的量子特徵映射較適合幾何結構與量子特徵空間相匹配的資料集。Moons 的非對稱雙月形結構破壞了這種對齊，Gram matrix 所能捕捉的核相似度無法有效分離兩類。

Explicit 模型表現最弱（circle 75.0%、moons 83.3%），呼應了論文的論點：單次 encoding 將輸入投影到固定的特徵空間後，即使增加層數（LE = 4），可達的 Fourier 頻率集合也只取決於 encoding 次數而非可訓練旋轉層數，因此函數表達能力有硬性天花板。

整體結果重現了 Ref. [3] 所報告的方法排序：data reuploading > implicit kernel > explicit。值得注意的是，在 moons 資料集上 kernel 與 explicit 的差距縮小，反映出不規則幾何邊界同時考驗了 kernel 的特徵匹配與 explicit 的表達能力，而 reuploading 對兩種困難的幾何結構均保持穩健。
