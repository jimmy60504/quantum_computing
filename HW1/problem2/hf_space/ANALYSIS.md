# Problem 2 分析

## 題目背景

這題要比較三種主流的量子機器學習（QML）分類方法，在兩個二元分類資料集上的表現。題目的核心問題是：**量子電路要怎麼「看」輸入資料？** 三種方法對這個問題的回答完全不同。

- **Explicit**：一開始 encode 一次，之後全靠 variational block 去學
- **Kernel**：完全不訓練，用量子電路定義一個 inner product，然後丟給 classical SVM
- **Reuploading**：每一層都重新把資料 encode 進去，讓量子態隨資料一起演化

這三種方法的比較是現在 QML 領域真正在研究的核心問題之一，不是 toy demo。

---

## 資料集

兩個資料集都是 2D 二元分類，200 個點，70/30 train/test split，所有特徵都經過 `StandardScaler` 正規化。

### Circle

```
Label = 1  if  ‖x‖ < r,  where r = √(2/π) ≈ 0.798
```

在 `[−1, 1]²` 上均勻取樣。這個半徑不是隨便選的：圓的面積 = πr² = 2，正方形面積 = 4，所以剛好一半點在圓內，一半在外，**類別完全平衡**。

邊界是一條完美的圓弧，是光滑的非線性邊界，但在概念上非常乾淨。對量子模型來說，這是個幾何測試：你的表示空間能不能把「遠離原點」跟「靠近原點」分開？

### Moons

sklearn 的 `make_moons(noise=0.1)`。兩個半月形互相交叉，邊界是非凸的，沒有辦法直接用一條曲線分開，需要兩個獨立的局部決策。這對所有方法都是更硬的挑戰。

---

## 三個方法

### 方法一：Explicit Quantum Model

**架構概念**：先把資料 encode 進量子態，再用一組 variational block 去轉動，最後量測。

**電路（L 層）**：

```
q0: ─[RX(x₀)]──[RY(θ)]─[RZ(θ)]─●──[RY(θ)]─[RZ(θ)]─●── ··· ─⟨Z₀⟩
                                  │                   │
q1: ─[RX(x₁)]──[RY(θ)]─[RZ(θ)]─X──[RY(θ)]─[RZ(θ)]─X── ···
     ↑ encode  └──────── layer 1 ─────────┘└──── layer 2 ────┘
      once
```

- **Encoding**：固定在第 0 層之前，`RX(x₀)` 和 `RX(x₁)` 各做一次
- **每層**：每個 qubit 做 `RY(θ)` + `RZ(θ)`，然後一個 `CNOT(0→1)` 負責 qubit 間糾纏
- **輸出**：量測 `⟨Z₀⟩`，映射為 class-1 的機率：`p = (⟨Z₀⟩ + 1) / 2`
- **Optimizer**：Adam，BCE loss
- **可訓練參數**：`4L`（L 層 × 2 qubits × 2 角度）

**關鍵限制**：資料只在最前面進入量子電路一次，之後 variational block 完全不再「看到」原始資料。這代表整個電路的表達能力受到初始 encoding 的 Hilbert space 嚴格限制。電路能學的邊界形狀，完全取決於 `RX(x₀), RX(x₁)` 初始化後的量子態空間。

---

### 方法二：Implicit Quantum Kernel (IQK)

**架構概念**：不訓練量子電路，而是用電路定義一個 **kernel function**，讓兩點 xᵢ, xⱼ 的「量子相似度」決定分類邊界。Classical SVM 負責找最優超平面。

**Feature map S(x)**：

```
q0: ─[H]─[RZ(x₀)]──────────────
                   ↘
q1: ─[H]─[RZ(x₁)]─[CNOT]─[RZ(x₀·x₁)]─
```

`H ⊗ H` 先把兩個 qubit 放進均勻疊加態，`RZ` 旋轉對應到資料的每個維度，`CNOT + RZ(x₀·x₁)` 加入交叉項，讓 kernel 能捕捉 x₀ 和 x₁ 的乘積關係。

**Kernel 定義**：

```
k(xᵢ, xⱼ) = |⟨0 | S†(xᵢ) S(xⱼ) | 0⟩|²
           = Pr( measure |00⟩ after S†(xᵢ)·S(xⱼ) )
```

把 S(xⱼ) 先做完，再做 S(xᵢ) 的反轉，量測回到 `|00⟩` 的機率。兩點「越相似」，機率越高，kernel 值越大。

**訓練流程**：
1. 算出訓練集完整的 `n×n` kernel matrix（需要 n² 次電路評估）
2. 把這個 matrix 丟給 `SVC(kernel="precomputed")` 做訓練
3. Inference 時只對測試點跟 **support vectors** 計算 kernel（不是全部訓練點）

**可訓練參數**：**0 個量子參數**。整個 feature map 是固定的，不做任何梯度更新。模型容量來自 SVM 的 dual coefficients，其數量等於 support vector 的數量。

**複雜度指標**：`count_model_complexity()` 回傳 kernel evaluations 的次數（訓練時 n²，推論時 n × n_sv）。

---

### 方法三：Data Reuploading Classifier

**架構概念**：每一層都重新把資料 encode 進去，讓資料和量子態深度交織。每次 encode 的 scale 和 bias 都是可學的，讓模型能自動決定「這個 encoding 的重要性」。

**電路（L 層）**：

```
q0: ─[RX(x₀·s₀⁽¹⁾+b₀⁽¹⁾)]─[RY(θ)]─[RZ(θ)]─●─[RX(x₀·s₀⁽²⁾+b₀⁽²⁾)]─ ··· ─⟨Z₀⟩
                                               │
q1: ─[RX(x₁·s₁⁽¹⁾+b₁⁽¹⁾)]─[RY(θ)]─[RZ(θ)]─X─[RX(x₁·s₁⁽²⁾+b₁⁽²⁾)]─ ···
     └──────────────── layer 1 ────────────────┘└──── layer 2 ────────
```

- **每層 encode**：`RX(xᵢ · scale[l,i] + bias[l,i])`，scale 和 bias 都是可訓練的
- **每層旋轉**：同 explicit，`RY(θ)` + `RZ(θ)` per qubit
- **每層糾纏**：`CNOT(0→1)`
- **輸出**：`⟨Z₀⟩` → 機率（同 explicit）
- **Optimizer**：Adam，BCE loss
- **可訓練參數**：`8L`（L 層 × 2 qubits × (1 scale + 1 bias + 2 rotations)）

**為什麼 reuploading 更強？**

在 Explicit 模型裡，資料只出現一次，相當於電路只能使用固定頻率成分。Reuploading 每層都重新 inject，理論上等價於一個 truncated Fourier series：**層數越多，可以近似的頻率越高，決策邊界可以越複雜**。

Pérez-Salinas et al. (2020) 證明了單一 qubit 的 data reuploading 是 universal approximator，前提是層數夠多。

---

## 三種方法的本質差異

|  | Explicit | Kernel | Reuploading |
|---|---|---|---|
| **資料進入電路** | 一次（固定 encoding） | 隱式（定義相似度） | 每層一次 |
| **訓練方式** | 梯度下降 (Adam) | 無量子訓練（SVM） | 梯度下降 (Adam) |
| **可訓練量子參數** | 4L | 0 | 8L |
| **表達能力隨 L** | 有限增長 | 固定（取決於 feature map） | Fourier 頻率隨 L 線性增加 |
| **訓練複雜度** | O(n·L) per epoch | O(n²) 一次性 | O(n·L) per epoch |
| **推論複雜度** | O(L) per point | O(n_sv) per point | O(L) per point |

---

## 看 viewer 時的注意點

**Slider**：拖動 slider 可以看訓練過程中決策邊界是怎麼形成的。Explicit 和 Reuploading 會隨 epoch 演化，Kernel 的邊界是固定的（只在 epoch 0 計算一次，之後複用）。

**6 個 panel 的對比邏輯**：
- 同一 row（同個 dataset）：三種方法在相同資料上的表現差異
- 同一 column（同個方法）：同一方法面對不同幾何結構的適應能力

**Circle vs Moons 的預期差異**：Circle 的邊界光滑，相對容易；Moons 非凸，需要更高的表達能力。特別值得觀察 Kernel 在 Moons 上的邊界能不能捕捉非凸結構，以及 Explicit 在層數不夠時是否會失敗。

**Reuploading 的早期行為**：scale 初始化為 1、bias 初始化為 0，早期 epoch 的邊界可能很不穩定，後期才收斂成有意義的形狀。這個過程和 Explicit 的演化方式會明顯不同。

---

_訓練結果與具體數字將在跑完後補充到這裡。_
