# Data Reuploading 回歸分析

這題的重點不是把目標函數硬背下來，而是理解一個小型 data reuploading circuit 在困難的 extrapolation split 下，究竟能從資料中推斷出什麼。這個 viewer 會同時用 3D 幾何、error map、loss curve 和 Fourier spectrum 來呈現這個過程。

## 這個網頁在展示什麼

這個頁面主要分成四個部分。左邊是不同超參數設定的結果表，方便直接比較 train / test MSE。中間上方的兩張 3D surface 分別對應 train domain 與 test domain，可以直觀看到模型學到的曲面形狀。右側的 error map 顯示誤差在平面上的分佈；下方的 log-scale loss curve 則讓我們觀察訓練過程中，batch loss 和 test MSE 是如何變化的。Reference 區塊另外提供 train/test split、目前電路圖，以及 Fourier spectrum 的輔助圖。

## 從畫面上最直接看到的現象

從 3D surface 可以直接看到，模型通常能很快把 train 區域貼得相當平順，甚至把 train MSE 壓到非常小；但同一張 surface 延伸到 test domain 之後，常常無法完整跟上真實曲面的下彎趨勢。error map 也會顯示 test 區域的誤差明顯大於 train 區域。另一方面，loss curve 常見的型態是 train loss 持續下降，但 test MSE 很早就進入平台區，這代表後期訓練帶來的改善多半只發生在 train domain。

## 對這些現象的第一層解讀

這表示這題對模型的要求並不是單純在已知資料中插值，而是要把左下角看到的局部幾何結構外推到右上角的未知區域。因此，即使模型已經在 train 上表現得很好，也不代表它真的理解了 target function 的整體結構。從結果來看，增加 qubits 或 layers 的確會提升表達能力，但效果不是單調的；有些較大的設定會把 train MSE 壓得更低，卻未必換來更好的 test surface。

## 為什麼這個 train/test split 特別難

train domain 位在左下角，目標函數在這裡只露出一小段高值區域的弧面；test domain 則位在右上角，那裡的曲面會明顯往下彎。這代表模型在 train 上很容易貼得很好，但到 test 時卻要把局部幾何外推出去，因此這題本質上更像 extrapolation，而不是一般的 interpolation。

## 這個電路實際上在做什麼

每一筆輸入 `x1, x2` 會先經過一個很小的 classical linear layer，再用 `tanh` 壓到較穩定的範圍，最後轉成量子電路要吃的旋轉角度。每一層都會把同一組角度重新注入到 qubit 上，先做 `RY` / `RZ` 編碼，再做 `CNOT` 糾纏，最後接上可訓練的 `Rot` gate。也就是說，這個模型學的是一種 hybrid representation：前後兩端由 classical layer 做介面轉接，中間則由量子電路提供結構化的非線性特徵。

## 這個模型其實不知道什麼

電路從來沒有被明確告知 `sin(exp(x1) + x2)` 這個生成規則。它只看得到 train 區域的局部監督訊號，然後試著把那段幾何延伸到 test domain。這也是為什麼 test surface 有時看起來像是「已經很努力了」，卻還是沒辦法真正恢復 sinusoidal 結構：模型學到的是局部形狀，不是底層公式本身。

## 為什麼 Fourier spectrum 很重要

Fourier view 會從頻率角度解釋同一件事。當訓練後的模型主要只抓到低頻成分時，surface 看起來會平滑、合理，但仍然缺少目標函數裡比較尖銳或更複雜的 oscillatory structure。這和 data reuploading circuit 的理論分析一致：circuit depth 會影響模型可表示的 Fourier modes，因此把 target 和 model 的 spectrum 並排比較，可以幫助解釋為什麼有些設定即使 train loss 很低，test error 還是會停在某個平台附近。
