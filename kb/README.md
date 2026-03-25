# 知識庫（KB）

這個 KB 用來記錄量子計算研究方向，先聚焦在方向釐清，不急著排實驗。

## 結構

- `research_directions.md`：研究方向清單與核心假設
- `directions/`：特定方向的詳細筆記
- `experiment_plans.md`：實驗規劃區（目前先暫停使用）
- `compute_hosts.md`：遠端主機與運算環境筆記
- `qml_stack_notes.md`：QML 框架定位與未來 image 策略
- `templates/research_direction_template.md`：研究方向模板
- `templates/experiment_plan_template.md`：實驗模板（方向確定後再使用）

## 建議流程（目前版本）

1. 先在 `research_directions.md` 記下候選方向。
2. 對有潛力的方向，建立 `directions/` 詳細筆記。
3. 等方向收斂後，再啟用 `experiment_plans.md`。

## 狀態定義

- 方向狀態：`想法`、`探索中`、`聚焦中`、`進行中`、`暫停`

## 快速開始

```bash
cp kb/templates/research_direction_template.md kb/directions/RD-001.md
```
