# Problem 1 分析

## 題目背景

這題的目標函數是 `f(x1, x2) = sin(exp(x1) + x2)`。這不是普通的回歸，而是一個**外推問題**：訓練集在 `[0, 0.5]²`（左下角），測試集在 `[0.5, 1.0]²`（右上角），兩者沒有重疊。

![Train/test split overview](./assets/problem1_data_overview.png)

模型必須在左下角學到曲面的結構，然後把這個結構推到從未見過的右上角。單純把 train loss 壓低沒有意義——判斷標準是能不能在 test domain 還原正確的曲率和相位。

這題有一個非常乾淨的量子解。因為 `sin(a + b) = ⟨Z⟩` 可以由一個 qubit 在同一旋轉軸上連續接收 `a` 和 `b` 再補上 `-π/2` 的相位移來實現，所以：

```
q0: ─[RY(exp(x1))]─[RY(x2)]─[RY(-π/2)]─⟨Z₀⟩
     ↑ 同軸累加角度        ↑ phase offset
```

這條路徑幾乎直接在量子電路裡編碼了 `sin(exp(x1) + x2)`。整個 viewer 的設計都圍繞這個 exact 解，沿著「一步一步放鬆先驗」的方向展開。

---

## 資料集

- **目標函數**：`f(x1, x2) = sin(exp(x1) + x2)`
- **訓練域**：`[0.0, 0.5] × [0.0, 0.5]`，均勻取樣
- **測試域**：`[0.5, 1.0] × [0.5, 1.0]`，均勻取樣
- **損失函數**：MSE

exp(x1) 的存在讓這題比一般的 sin 回歸難很多——模型要學的不只是 `sin(a + b)` 的加法結構，還要先自己恢復 `exp(x1)` 這個內部表示。這是貫穿整個 viewer 最核心的問題。

---

## 九個模型

模型從 exact 解出發，一步一步放鬆先驗，分成三個家族。

---

### Exact 家族（1 qubit，1 layer）

這三個模型使用固定的 `encode_features(x) = [exp(x1), x2]`，即模型已知 `exp(x1)` 是正確的特徵表示。

#### `quantum_exact`

最乾淨的 exact 解。全部固定，沒有任何可訓練參數：

```
q0: ─[RY(exp(x1))]─[RY(x2)]─[RY(-π/2)]─⟨Z₀⟩
```

這不是訓練出來的，而是直接把 `sin(a + b) = ⟨Z⟩` 的恆等式硬編碼進去。**final test MSE ≈ 7.34e-15**，幾乎是數值誤差。

#### `phase_learnable`

把 `-π/2` 改成可學習的 phase shift（初始化為 `-π/2`）：

```
q0: ─[RY(exp(x1))]─[RY(x2)]─[RY(φ)]─⟨Z₀⟩
     φ 初始化為 -π/2，可訓練
```

**可訓練參數：1**（φ）。這是最小幅度的放鬆：確認只要 encoding 對，稍微擾動 phase 也能快速恢復。**final test MSE ≈ 5.77e-11**。

#### `scaled_exact`

在 `phase_learnable` 的基礎上，讓 `exp(x1)` 和 `x2` 的 scale / bias 也可以學：

```
q0: ─[RY(s₁·exp(x1)+b₁)]─[RY(s₂·x2+b₂)]─[RY(φ)]─⟨Z₀⟩
     s₁,b₁,s₂,b₂ 各自初始化為 (1,0)，φ 初始化為 -π/2
```

**可訓練參數：5**（s₁, b₁, s₂, b₂, φ）。確認在給定正確特徵的前提下，這個小幅自由度不會破壞解。**final test MSE ≈ 2.06e-10**。

---

### Same-axis Reupload 家族（1 qubit，多 layers）

這個家族保留「同一 qubit、同一旋轉軸（Y 軸）」的 backbone，但用多個 block 重複 encoding，讓模型自己學習如何組合。每個 block 的 scale、bias 和 phase shift 都是可訓練的。

**每個 block 的結構**：

```
q0: ─[RY(s₁⁽ˡ⁾·f₁+b₁⁽ˡ⁾)]─[RY(s₂⁽ˡ⁾·f₂+b₂⁽ˡ⁾)]─[RY(φ⁽ˡ⁾)]─ ···
```

其中 f₁, f₂ 因版本而異（見下表），`l` 是 block index。

| 模型 | 輸入特徵 f₁, f₂ | 可訓練參數（L 層）|
|---|---|---|
| `same_axis_reupload` | `exp(x1), x2` | 4L（per-block scale, bias, phase） |
| `same_axis_raw` | `x1, x2` | 4L |
| `same_axis_poly` | `poly(x1,θ), x2`（3 次多項式係數可學） | 4L + 4L（poly） |

`same_axis_reupload` 保留 `exp(x1)` 先驗，是「真正的 data reuploading 版本」：模型自己決定如何在多個 block 中混合和組合這兩個特徵。`same_axis_raw` 和 `same_axis_poly` 則測試在不提供 `exp(x1)` 的情況下，模型能不能自己近似這個非線性變換。

#### 代表性結果（L=2，10 epochs）

- `same_axis_reupload`：best test MSE ≈ **4.91e-3**
- `same_axis_poly`：best test MSE ≈ **2.88e-2**
- `same_axis_raw`：best test MSE ≈ **3.22e-2**

`same_axis_reupload` 的誤差比 `same_axis_raw` 和 `same_axis_poly` 小一個數量級，說明 `exp(x1)` 這個先驗的信息量非常高，靠學習近似它代價很大。

---

### 2-qubit 家族

用兩個 qubit 測試不同的「變數組合」方式。

#### `twoqubit_no_reupload`

利用和角公式：把 `exp(x1)` 和 `x2` 分別 encode 到兩個 qubit，再量測跨 qubit 的 correlator：

```
q0: ─[RY(exp(x1))]─⟨X₀Z₁⟩
q1: ─[RY(x2)]─────⟨Z₀X₁⟩
```

輸出 = `w₁·⟨X₀Z₁⟩ + w₂·⟨Z₀X₁⟩`（w₁, w₂ 固定為 `[1,0]` 的解析解）

因為 `⟨X₀Z₁⟩ = sin(exp(x1))·cos(x2)`、`⟨Z₀X₁⟩ = cos(exp(x1))·sin(x2)`，
兩者相加就是 `sin(exp(x1) + x2)`，完全不需要 reupload。**final test MSE ≈ 5.91e-15**。

#### `twoqubit_raw_no_reupload`

把 `exp(x1)` 先驗拿掉，改用 raw `x1, x2`，加上 entangling block 後量測四個 observable：

```
q0: ─[RY(x1)]─●─[Rot(α₀,β₀,γ₀)]─⟨Z₀⟩, ⟨X₀Z₁⟩
              │
q1: ─[RY(x2)]─X─[Rot(α₁,β₁,γ₁)]─⟨Z₁⟩, ⟨Z₀X₁⟩
```

輸出 = 四個 observable 的可學習線性組合。**best test MSE ≈ 2.03e-1**——遠比 `twoqubit_no_reupload` 差。2 qubits 和 entanglement 並不足以讓模型自動學出 `exp(x1)` 這個非線性變換。

#### `same_axis_twoqubit`

把 same-axis reupload 架構推廣到 2 qubits。每個 block 中，q0 和 q1 各自做 same-axis reupload，然後用 `CNOT(1→0)` 建立 qubit 間的糾纏，最後對 `⟨Z₀⟩` 和 `⟨Z₁⟩` 做可學習線性組合輸出。

---

## 各模型的本質差異

| 模型 | `exp(x1)` 先驗 | Reupload | Qubits | 可訓練參數 |
|---|---|---|---|---|
| `quantum_exact` | ✓（固定） | — | 1 | 0 |
| `phase_learnable` | ✓ | — | 1 | 1 |
| `scaled_exact` | ✓ | — | 1 | 5 |
| `same_axis_reupload` (L層) | ✓ | ✓ | 1 | 4L |
| `same_axis_raw` (L層) | ✗（raw x1） | ✓ | 1 | 4L |
| `same_axis_poly` (L層) | 近似（3次多項式） | ✓ | 1 | 8L |
| `twoqubit_no_reupload` | ✓ | ✗ | 2 | 0 |
| `twoqubit_raw_no_reupload` | ✗ | ✗ | 2 | ~10 |
| `same_axis_twoqubit` (L層) | ✓ | ✓ | 2 | 10L |

**核心結論：決定模型表現的不是 qubit 數或 entanglement，而是 `exp(x1)` 先驗有沒有保住。** 只要提供正確的特徵表示，無論是 1-qubit reupload 還是 2-qubit no-reupload，都能非常接近 exact 解；反之，拿掉 `exp(x1)` 之後，即使增加模型複雜度，誤差都會顯著上升。

---

## 頻譜的角色

Fourier spectrum 用來確認模型有沒有學到正確的主頻結構。

`quantum_exact` 的頻譜和 target 幾乎完全重合，與近似零誤差一致。`same_axis_reupload` 的主頻位置已經沿著正確方向靠近 target，但幅度還沒完全對齊——這和 3D surface 上看到的「曲率大致對、但還有殘差」是同一件事的兩個角度。

![Circuit structure](./assets/problem1_circuit.png)

---

## 看 viewer 時的注意點

**建議的觀看順序**：
1. 先看 `quantum_exact`，把它當作結構參考。
2. 看 `phase_learnable` → `scaled_exact`，確認小幅放鬆後 exact 解是否穩定。
3. 看 `same_axis_reupload`，觀察 data reuploading 版本學到的曲面離參考解差多少。
4. 對比 `same_axis_raw` 和 `same_axis_poly`——拿掉 `exp(x1)` 先驗後，誤差如何放大。
5. 最後看 `twoqubit_no_reupload` vs `twoqubit_raw_no_reupload`：同樣是 2 qubits，有沒有 `exp(x1)` 的差距就是結論的直接呈現。

**3D surface**：slider 拖動可以看訓練過程中 test domain 的曲面是怎麼逐漸成形的。特別注意在沒有 `exp(x1)` 先驗的模型中，曲率能不能在 test domain 的右上角正確彎折。

**Train vs Test 差距**：Train domain 對所有模型都相對容易擬合。真正的考驗是 test domain——如果一個模型 train MSE 很低但 test MSE 高，表示它只是把 train domain 記住了，沒有學到正確的結構。

---

_訓練結果與具體數字請以 runtime export 為準；本頁面的數字來自代表性實驗。_
