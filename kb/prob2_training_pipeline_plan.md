# HW1 Problem 2 — Training Pipeline Plan

## Context

Problem 2 要比較三種 QML 分類方法在兩個 dataset 上的表現，並把整個訓練過程顯示在 HF Space viewer 上。現有 scaffold 已定義好 `QuantumClassifier` Protocol、`DatasetBundle`、`BenchmarkResult` 等介面，但三個模型的實作和訓練流程都還空白。

目標：建一個完整 pipeline，從模型實作 → 訓練 + MLflow → per-step 匯出 → viewer 顯示。

---

## File Structure (新增/修改)

```
HW1/problem2/
  core/
    __init__.py
    config.py                  # Prob2Config dataclass
    training.py                # 主訓練循環 + MLflow + viewer export
    viewer_io.py               # write_viewer_export(), update_viewer_manifest()
    models/
      __init__.py
      explicit.py              # ExplicitQuantumClassifier
      kernel.py                # QuantumKernelClassifier
      reuploading.py           # DataReuploadingClassifier
  train.py                     # CLI entry point (argparse → train())
  scripts/
    gx10_hw1_prob2_pipeline.sh # gx10 full pipeline script
```

---

## 1. 三個模型實作

全部用 PennyLane，遵守現有 `QuantumClassifier` Protocol（fit, predict, decision_function, count_model_complexity）。

### Explicit Quantum Model (`core/models/explicit.py`)
- 2 qubits，固定 angle encoding `RX(x0), RX(x1)`
- L 層 variational block：`RY(θ) + RZ(θ)` per qubit + `CNOT(0,1)`
- 測量 `⟨Z₀⟩`，映射 `(expval+1)/2` 為 class-1 機率
- 用 `qml.qnn.TorchLayer` 包成 `nn.Module`，`Adam` optimizer
- Trainable params: 4L

### Implicit Quantum Kernel (`core/models/kernel.py`)
- Feature map: `H` + `RZ(x0), RZ(x1)` + `CNOT` + `RZ(x0*x1)` 交互項
- Kernel: `k(xi,xj) = |⟨0|S†(xi)S(xj)|0⟩|²` 用 `qml.kernels.kernel_matrix()`
- 餵進 `sklearn.svm.SVC(kernel="precomputed")`
- 無可訓練量子參數，只算 kernel evaluations
- 一次 fit，沒有 epoch loop

### Data Reuploading (`core/models/reuploading.py`)
- 2 qubits，L 層，每層重新 encode：`RX(x_i * w_scale + w_bias)`
- 每層後接 `RY(θ) + RZ(θ)` + `CNOT`
- 測量 `⟨Z₀⟩`，同 explicit 映射
- Trainable params: 8L

---

## 2. Config (`core/config.py`)

```python
@dataclass
class Prob2Config:
    # General
    seed: int = 11224001
    n_samples: int = 200
    epochs: int = 50
    learning_rate: float = 0.05
    batch_size: int = 32

    # Method-specific
    num_qubits: int = 2
    num_layers_explicit: int = 4
    num_layers_reuploading: int = 4

    # Viewer export
    boundary_grid_size: int = 50      # 50×50 decision boundary grid
    viewer_export_every: int = 5      # export step every N epochs

    # Paths
    viewer_export_path: str = "HW1/problem2/hf_space/runtime"

    # MLflow
    tracking_uri: str = ""
    experiment_name: str = "hw1-problem2-qml-classifiers"
    run_name: str = ""
```

---

## 3. 訓練流程 (`core/training.py`)

單一 `train(config)` 函式，在一個 MLflow run 內完成全部工作：

```
1. 載入 circle + moons datasets
2. 初始化三個模型（每個 dataset 各一份 → 6 個模型實例）
3. Kernel 方法先一次 fit 完，預算好 decision boundary（固定不變）
4. Epoch loop (1..epochs):
   a. 對 explicit 和 reuploading 各在兩個 dataset 上訓練一個 epoch
   b. 計算 train_acc, test_acc, loss
   c. MLflow log: per-epoch metrics（6 個 method×dataset 的 acc/loss）
   d. 每 viewer_export_every 個 epoch：
      - 計算 explicit + reuploading 的 decision boundary
      - 組合成 timeline_step（含全部 6 個 boundary + scatter + accuracies）
      - 寫入 viewer JSON
5. 最終匯出 + update manifest
6. MLflow log artifacts
```

### MLflow logging

**Params**: seed, n_samples, epochs, lr, num_qubits, layers_explicit, layers_reuploading, boundary_grid_size

**Per-epoch metrics** (keyed by global step = epoch):
- `explicit/circle/train_acc`, `explicit/circle/test_acc`, `explicit/circle/loss`
- 同理其他 5 個 method×dataset 組合
- `kernel/*` 只在第一步 log 一次（不變）

**Final summary**:
- `best_test_acc`, per-method per-dataset final accuracies
- Training time per method

**Artifacts**:
- viewer_data.json, viewer_manifest.json → `artifact_path="viewer"`

---

## 4. Viewer Export Format (`core/viewer_io.py`)

### viewer_data.json

```json
{
  "title": "QML Classifier Explorer",
  "subtitle": "QCAA HW1 Problem 2 classification results",
  "status": "trajectory export",
  "experiment": { ... },
  "assets": { "datasets_overview": "assets/preview_datasets.png" },
  "grid": { "resolution": 50 },
  "scatter": {
    "circle": [{"x": 0.1, "y": 0.2, "label": 0}, ...],
    "moons":  [...]
  },
  "timeline_steps": [
    {
      "label": "Epoch 5",
      "epoch": 5,
      "global_step": 5,
      "train_acc": 0.82,
      "test_acc": 0.78,
      "train_loss": 0.41,
      "test_loss": 0.48,
      "accuracies": {
        "explicit":    {"circle": 0.85, "moons": 0.80},
        "kernel":      {"circle": 0.90, "moons": 0.88},
        "reuploading": {"circle": 0.83, "moons": 0.79}
      },
      "boundaries": {
        "explicit": {
          "circle": {"x": [...], "y": [...], "z": [[...]]},
          "moons":  {"x": [...], "y": [...], "z": [[...]]}
        },
        "kernel":      { ... },
        "reuploading": { ... }
      }
    },
    ...
  ]
}
```

### viewer_manifest.json

```json
{
  "title": "QML Classifier Runs",
  "default_run": "q2-l4-e50",
  "runs": [
    {
      "id": "q2-l4-e50",
      "label": "2 qubits, 4 layers, 50 epochs",
      "path": "./runtime/q2-l4-e50.json",
      "steps": 10,
      "num_qubits": 2,
      "num_layers_explicit": 4,
      "num_layers_reuploading": 4,
      "best_test_acc": 0.92,
      "methods": ["explicit", "kernel", "reuploading"]
    }
  ]
}
```

---

## 5. CLI Entry Point (`train.py`)

```
python HW1/problem2/train.py \
  --epochs 50 \
  --layers-explicit 4 \
  --layers-reuploading 4 \
  --learning-rate 0.05 \
  --tracking-uri http://gx10-mlflow-server:5001 \
  --run-name q2-l4-e50
```

---

## 6. gx10 Pipeline Script (`scripts/gx10_hw1_prob2_pipeline.sh`)

```bash
# 1. Start MLflow server
./scripts/gx10_mlflow_server.sh start

# 2. Run configurations
for config in "4 4" "6 6" "2 2" "4 8"; do
  read le lr <<< "$config"
  RUN_NAME="q2-le${le}-lr${lr}-e50"
  bash scripts/gx10_run_py.sh HW1/problem2/train.py \
    --layers-explicit $le --layers-reuploading $lr \
    --run-name $RUN_NAME \
    --tracking-uri http://gx10-mlflow-server:5001
done
```

需要 `--network gx10-mlflow` 加到 `gx10_run_py.sh` 的 `GX10_DOCKER_NETWORK` 環境變數。

---

## 7. Verification

1. **Unit**: 在 gx10 上先跑 1 epoch、1 layer 確認三個模型都能 fit/predict
2. **Export**: 確認 viewer_data.json 的 timeline_steps 格式被 HF Space viewer 正確讀取
3. **MLflow**: 確認 metrics 出現在 MLflow UI（`ssh -L 5001:localhost:5001 gx10`）
4. **Viewer**: 本地開 index.html 確認 6 個 decision boundary 都有畫出來

---

## 8. 實作順序

1. `core/config.py` — Config dataclass
2. `core/models/reuploading.py` — 最熟悉、有參考
3. `core/models/explicit.py` — 類似 reuploading 但更簡單
4. `core/models/kernel.py` — 完全不同的 pattern (SVM)
5. `core/training.py` — 訓練循環 + MLflow + boundary 計算
6. `core/viewer_io.py` — JSON export
7. `train.py` — CLI
8. `scripts/gx10_hw1_prob2_pipeline.sh` — pipeline
9. Test on gx10 with small config, iterate
