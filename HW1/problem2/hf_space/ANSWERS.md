# Problem 2 — 作答

隨機種子：11224001。所有實驗使用 2 個量子位元，n_samples = 200（140 訓練 / 60 測試），訓練 50 個 epoch，學習率 0.05，批次大小 32。

---

## (a) 重現 Fig. 6 — Circle 資料集

互動介面中呈現了三種方法在 circle 資料集上的訓練曲線，定性上重現了 Ref. [3] Fig. 6 的結果。Implicit kernel 和 data reuploading 兩種方法均在 50 個 epoch 內收斂至超過 96% 的測試準確率，而 explicit 模型停滯在約 77%。Kernel 方法以單次 SVM 擬合完成訓練（不需迭代 epoch），因此其曲線從一開始就是水平的。結果與文獻中報告的方法排序一致。

## (b) 決策邊界

3 × 2 的決策邊界圖格顯示於互動介面中。各方法特性如下：

- **Explicit**：在 circle 資料集上產生近似圓形但不夠精確的邊界；在 moons 資料集上邊界較平滑，但在兩側末端略有偏差。
- **Kernel**：在 circle 上形成較緊密的圓形邊界；在 moons 上邊界較不規則，反映出核方法整體擬合的特性。
- **Data Reuploading**：在兩個資料集上均產生最銳利的決策邊界，最貼近資料的真實分布。

## (c) 比較表

實驗設定：`q2-le4-lr4-e50`（explicit encoding layers LE = 4，reuploading layers LR = 4）

| 方法 | 資料集 | 測試準確率 | 可訓練參數 / 核函數計算次數 | 訓練時間 |
|---|---|---|---|---|
| Explicit | Circle | 76.7 % | 16 個參數 | ≈ 20 s |
| Explicit | Moons | 88.3 % | 16 個參數 | ≈ 20 s |
| Implicit Kernel | Circle | 96.7 % | 19,600 次核計算 | ≈ 10 s |
| Implicit Kernel | Moons | 85.0 % | 19,600 次核計算 | ≈ 10 s |
| Data Reuploading | Circle | 98.3 % | 32 個參數 | ≈ 30 s |
| Data Reuploading | Moons | 98.3 % | 32 個參數 | ≈ 30 s |

核函數計算次數 = 140 × 140 = 19,600（完整訓練 Gram matrix）。Implicit kernel 的訓練時間不隨 epoch 數增加，因為 SVM 僅需單次擬合。

## (d) 討論

Data reuploading 在兩個資料集上均達到最高且最一致的準確率（各 98.3%），原因在於將資料重複編碼穿插於可訓練旋轉之間，大幅擴展了相同參數數量下的函數表達能力。Implicit kernel 方法在對稱的 circle 資料集上表現良好（96.7%），但在 moons 上明顯下降至 85.0%，與 Ref. [3] 的觀察一致——kernel 的量子特徵映射較適合幾何結構與量子特徵空間相匹配的資料集。Explicit 模型表現最弱（circle 76.7%、moons 88.3%），呼應了論文的論點：單次編碼的表達能力有限，即便增加層數也難以有效改善。值得注意的是，reuploading 對非對稱的 moons 幾何結構保持穩健，而 kernel 則出現明顯退化，顯示 reuploading 的迭代資料重注入對不規則邊界具有更好的歸納偏置。整體結果重現了 Ref. [3] 所報告的方法排序：data reuploading > implicit kernel > explicit，但在 moons 資料集上差距縮小，反映出資料集難度的提升使三者趨於平衡。
