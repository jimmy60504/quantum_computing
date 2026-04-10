# Problem 3 分析

## 題目背景

這題要在 CIFAR-10 上比較兩種分類器的表現：一個用傳統 MLP 作為分類頭，一個用參數化量子電路（PQC）替換分類頭。問題的核心是：**在現實規模的影像分類任務上，量子電路作為分類頭，能不能跟同等規模的古典網路競爭？**

兩個模型共用同一個 CNN backbone 做特徵提取，只有最後的分類頭不同。Backbone 解凍（unfrozen）一起訓練，確保兩邊的特徵提取都能適應分類任務。

---

## 資料集

CIFAR-10：60,000 張 32×32 RGB 影像，10 個類別，50,000 train / 10,000 test。是評估輕量分類頭的合適 benchmark。

---

## 共用 backbone：SimpleCNNBackbone（56,320 個可訓練參數）

```
Input: 3×32×32
Block 1: Conv2d(3→32, 3×3) → BN → ReLU → MaxPool2d(2)   → 32×16×16
Block 2: Conv2d(32→64, 3×3) → BN → ReLU → MaxPool2d(2)   → 64×8×8
Block 3: Conv2d(64→128, 3×3) → BN → ReLU → MaxPool2d(2)  → 128×4×4
         AdaptiveAvgPool2d(1) → Flatten                    → 128-D
         Linear(128, 256)                                  → 256-D
```

---

## 兩個模型

### 模型一：CNN + MLP Baseline

分類頭為單層線性分類器：`Linear(256, 10)`，head 參數 2,570。總參數 **58,890**。

### 模型二：CNN + QNN Hybrid（4 qubits × 2 layers）

```
256-D features
  → Linear(256, 4)      [pre layer, 1,028 params]
  → tanh(·) × π         [scale to [-π, π]]
  → RY(xᵢ) for i=0..3   [angle encoding]
  → [RY(θ) + RZ(θ) + CNOT ring] × 2 layers   [variational, 16 quantum params]
  → [⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩]   [4 PauliZ expectations]
  → Linear(4, 10)        [post layer, 50 params]
  → logits
```

Head 參數：1,028 + 16 + 50 = **1,094**。總參數 **57,414**。

---

## 實驗結果

### 最終測試準確率（20 epochs）

| 模型 | 最佳 Test Acc | 可訓練參數 | 訓練時間 |
|------|-------------|----------|---------|
| CNN + MLP | **76.81%** | 58,890 | 59 s |
| CNN + QNN | 61.07% | 57,414 | 18,250 s（≈5.1h） |

QNN 每 epoch 耗時約 912 秒，是 MLP（≈3 s/epoch）的 **311 倍**。根本原因是量子電路模擬器無法 batch 平行，每個樣本必須串行執行 QNode，即使使用 GPU（lightning.gpu）也是如此，因為 PennyLane 的 autograd 不支援 vmap over quantum circuits。

### 訓練曲線趨勢

| Epoch | MLP test | QNN test |
|-------|---------|---------|
| 1 | 52.95% | 28.19% |
| 5 | 68.26% | **45.24%**（大幅跳躍） |
| 10 | 73.13% | 55.60% |
| 15 | 75.89% | 59.80% |
| 20 | 76.81% | 61.07% |

- **MLP** 從第一 epoch 就學得很好（52.95%），收斂平滑，第 15 epoch 後幾乎飽和。
- **QNN** 前 4 個 epoch 表現非常差（28%→36%），epoch 5 出現明顯跳躍（45%），此後持續緩慢爬升。第 20 epoch 仍未完全收斂（epoch 19 train=60.3%、epoch 20 train=60.7%），顯示可能還有上升空間但已趨於平緩。

QNN 前期的低迷可能反映了量子電路的 **barren plateau** 現象：隨機初始化的 PQC 梯度在高維 Hilbert space 中指數衰減，導致訓練初期幾乎無法學習，直到梯度找到有效方向才突破。

### 混淆矩陣分析（Epoch 20）

**MLP 各類別 accuracy：**

| 類別 | Acc | 主要誤分 |
|------|-----|---------|
| airplane | 78.7% | ship |
| automobile | 86.9% | — |
| bird | **64.9%** | cat, deer |
| cat | **58.0%** | dog, deer |
| deer | 75.5% | horse |
| dog | 69.3% | cat |
| frog | 85.8% | — |
| horse | 75.4% | deer |
| ship | 89.1% | — |
| truck | 85.0% | automobile |

**QNN 各類別 accuracy：**

| 類別 | Acc | 主要誤分 |
|------|-----|---------|
| airplane | 64.5% | ship |
| automobile | 77.0% | truck |
| bird | **28.2%** | deer, frog |
| cat | **13.5%** | dog（嚴重崩潰） |
| deer | 50.3% | horse |
| dog | 69.3% | cat |
| frog | 79.3% | — |
| horse | 74.3% | deer |
| ship | 73.8% | airplane |
| truck | 80.4% | automobile |

QNN 在視覺上相似的細粒度類別（bird、cat）表現嚴重崩潰。**Cat 僅達 13.5%**，主要被誤分為 dog，顯示 4 維的量子特徵空間根本無法區分這些需要高維特徵才能辨識的類別。相比之下，視覺特徵較鮮明的類別（frog、truck）QNN 表現還不錯（79%、80%）。

### 訓練 vs 測試準確率差距

QNN 的 train-test gap 幾乎為零（epoch 20：train 60.72%、test 61.07%），甚至 test 略高於 train。這意味著 QNN 完全沒有過擬合，反而是 **欠擬合**——電路表達能力不足以擬合訓練資料。4 個 qubit 限制了 QNN 的容量，Hilbert space 只有 2⁴ = 16 維。

MLP 也沒有明顯過擬合（epoch 20：train 75.17%、test 76.81%），因為 head 本身非常簡單（僅 2,570 params），主要容量來自共用的 backbone。

---

## 結論

在這個 unfrozen backbone、相近總參數量的設定下，QNN (4q×2l) 比 MLP 低 **15.74 個百分點**，且慢 311 倍。差距主要來自：

1. **特徵空間維度**：4-qubit QNN 只能在 4 維中處理 256-D 特徵，而 MLP 直接用 256-D 接 10-D；量子態的 16 維 Hilbert space 不足以表示複雜的視覺語義邊界。
2. **訓練動態**：QNN 前期 barren plateau 造成學習緩慢，同樣的 epoch 數下 QNN 還未收斂至最佳點。
3. **計算成本**：311× 的訓練時間差距讓 QNN 在相同運算預算下能跑的 epoch 數遠少於 MLP。

量子優勢在目前的嘈雜中等規模量子（NISQ）設備和模擬器上，對於經典機器視覺任務並不存在。
