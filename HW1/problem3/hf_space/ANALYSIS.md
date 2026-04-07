# Problem 3 分析

## 題目背景

這題要在 CIFAR-10 上比較兩種分類器的表現：一個用傳統 MLP 作為分類頭，一個用參數化量子電路（PQC）替換分類頭。問題的核心是：**在現實規模的影像分類任務上，量子電路作為分類頭，能不能跟同等規模的古典網路競爭？**

兩個模型共用同一個 CNN backbone 做特徵提取，只有最後的分類頭不同。這樣的設計隔離了兩者的差異——backbone 的特徵品質對兩邊是一樣的，差別只在頭部的表達方式。

---

## 資料集

CIFAR-10：60,000 張 32×32 RGB 影像，10 個類別（airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck），50,000 train / 10,000 test。這是一個標準的多分類 benchmark，在古典模型上已有很充分的對照基準，適合拿來評估量子方法在「真實任務」上的表現。

---

## 共用 backbone：SimpleCNNBackbone

兩個模型共用的 CNN 特徵提取器，輸出 256-D 向量。

**架構**：

```
Input: 3×32×32

Block 1: Conv2d(3→32, 3×3) → BN → ReLU → MaxPool2d(2)   → 32×16×16
Block 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool2d(2)   → 64×8×8
Block 3: Conv2d(64→128, 3×3) → BN → ReLU → MaxPool2d(2)  → 128×4×4
         AdaptiveAvgPool2d(1) → Flatten                    → 128-D
         Linear(128, 256)                                  → 256-D
```

這個 backbone 是刻意設計成「夠輕量但不太弱」的特徵提取器。三個 conv block 配合 batch norm，足以捕捉 CIFAR-10 的低階視覺特徵（邊緣、紋理），但不會強到讓分類頭完全不重要。

---

## 兩個模型

### 模型一：CNN + MLP Baseline

**分類頭**：兩層隱藏層（各 128 維），加 Dropout(0.2)，最後接 10-way linear。

```
256-D features
  → Linear(256, 128) → ReLU → Dropout(0.2)
  → Linear(128, 128) → ReLU → Dropout(0.2)
  → Linear(128, 10)
  → logits
```

**Head 參數量**：256×128 + 128 + 128×128 + 128 + 128×10 + 10 = **50,314**

這是一個非常標準的分類頭，也是大多數輕量 CNN 的預設選擇。Dropout 提供正則化，防止 backbone 輸出過擬合。

---

### 模型二：CNN + QNN Hybrid

**分類頭**：先用一個線性層把 256-D 壓縮到 num_qubits，進量子電路做 angle encoding + variational layers，量測後再用一個線性層映射到 10 個 logits。

```
256-D features
  → Linear(256, 8)   [pre layer]
  → tanh(·) × π      [scale to [-π, π]]
  → RY(xᵢ) for i=0..7   [angle encoding]
  → [RY(θ) + RZ(θ) per qubit → CNOT ring] × 4 layers   [variational]
  → [⟨Z₀⟩, ⟨Z₁⟩, …, ⟨Z₇⟩]   [8 PauliZ expectations]
  → Linear(8, 10)    [post layer]
  → logits
```

**量子電路（per sample）**：

```
q0: ─[RY(x₀)]─[RY(θ)]─[RZ(θ)]─●────────────────────────────X─ ··· ─⟨Z₀⟩
q1: ─[RY(x₁)]─[RY(θ)]─[RZ(θ)]─X─●───────────────────────── ─ ···
q2: ─[RY(x₂)]─[RY(θ)]─[RZ(θ)]───X─●──────────────────────── ─ ···
  ⋮                                    ⋮
q7: ─[RY(x₇)]─[RY(θ)]─[RZ(θ)]─────────────────────────────●─ ···
                └──────── layer 1 ──────────────────────────┘
```

CNOT 的連接方式是「環形」（ring entanglement）：qubit i → (i+1) % n，讓每一層都能在所有相鄰 qubit 之間建立糾纏，最後一個 qubit 回接第一個。

- **角度編碼**：`tanh(pre(features)) × π`，把 256-D features 壓縮並縮放到 `[-π, π]`，作為各 qubit 的 RY 旋轉角度
- **Variational 參數**：`(num_layers, num_qubits, 2)` = 4×8×2 = **64 個量子參數**，初始化為 `0.01 × randn`
- **後處理**：8 個 PauliZ 期望值（`[-1, 1]`）接 `Linear(8, 10)` 輸出 logits
- **梯度計算**：`default.qubit` + `backprop`（parameter-shift 的替代方案，通過模擬器直接反傳）

**Head 參數量**：
- pre: 256×8 + 8 = **2,056**
- 量子電路: 4×8×2 = **64**
- post: 8×10 + 10 = **90**
- 合計：**2,210**

---

## 兩個模型的比較

|  | CNN + MLP | CNN + QNN |
|---|---|---|
| **Backbone** | SimpleCNNBackbone (shared) | SimpleCNNBackbone (shared) |
| **Head 架構** | 2×Linear(256→128→10) | Linear(256→8) → PQC(8q, 4L) → Linear(8→10) |
| **Head 參數量** | ~50,314 | ~2,210 |
| **量子參數** | 0 | 64 |
| **Head 瓶頸維度** | 128 | 8（量子態空間） |
| **訓練方式** | Adam + CrossEntropy | Adam + CrossEntropy（同 backbone 一起更新） |
| **推論複雜度** | O(1) per sample | O(batch_size × num_qubits × num_layers)（模擬器逐樣本串行） |

**關鍵不對稱**：MLP head 有 5 萬多個參數，QNN head 只有 2 千多個。這個差距在設計上是刻意的：量子態空間的維度限制（8 個 qubit → 2⁸ = 256 維 Hilbert space），讓 PQC 的容量天花板遠低於 MLP。這題真正在測試的是：**在資源受限（參數量差 20 倍以上）的情況下，量子電路能不能靠 Hilbert space 的結構性優勢補回來？**

---

## 看 viewer 時的注意點

**Comparison table**：顯示兩個方法各自的最佳 test accuracy、參數量、訓練時間。這是最直接的比較點。

**Training curves**：紫色線（MLP）vs 橙色線（QNN），實線是 train acc，虛線是 test acc。拖 slider 可以看到哪個 epoch 的 confusion matrix，同時在曲線上會有一條垂直的 marker 對齊。

**Confusion matrices**：CIFAR-10 的類別不是等難的。bird/cat/deer 在視覺上相近，預期錯誤集中在這些類之間。注意觀察 QNN 的混淆矩陣是否出現系統性的對角線崩潰（表示某些類幾乎完全猜錯），還是均勻退化——兩者對應不同的失敗模式。

**訓練速度**：QNN 的每個 epoch 比 MLP 慢很多，因為量子電路目前是逐樣本串行模擬（batch loop 在 Python 層）。訓練時間的差距本身也是一個值得注意的實驗結果。

---

_訓練結果與具體數字將在跑完後補充到這裡。_
