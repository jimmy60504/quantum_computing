# Problem 1 說明

這個頁面整理了 Problem 1 的主要結果、訓練過程和幾張輔助圖，方便把不同設定放在同一個地方看。

## 題目背景

這一題要做的是回歸 `f(x1, x2) = sin(exp(x1) + x2)`，但資料不是從整個平面隨機抽，而是刻意把 train domain 放在 `[0.0, 0.5] x [0.0, 0.5]`，test domain 放在 `[0.5, 1.0] x [0.5, 1.0]`。也就是說，模型只能先在左下角看到一小塊區域，接著再拿去推右上角那塊沒有直接看過的區域，所以這題本身就帶有很明顯的 extrapolation 性質。

![Train/test split overview](./assets/problem1_data_overview.png)

左半邊是整個 target function 的熱度分布，右半邊把 train samples 和 test samples 分開標出來，所以可以很直觀看到 train 和 test 是怎麼被切開的，也更容易把這個 split 和後面看到的泛化表現連起來看。

## 先看這個頁面有什麼

左邊的 `Results` 會列出每一組設定的 train / test MSE，現在主要用來比較不同 qubits 和 layers 的表現。中間上方的兩張 3D 圖分別是 train domain 和 test domain 的預測曲面，右邊是對應的 error map。最下面的 loss 圖可以配合 slider 看每一個 step 的變化。左側 `Experiment` 卡片的上方則放了目前的電路圖，以及 Fourier spectrum。

如果只是第一次進來，最簡單的看法是：

1. 先在左邊挑一組設定。
2. 看上面的 train / test 3D 圖。
3. 再看右邊 error map 哪裡最亮。
4. 最後拖下面的 slider，看這個結果是怎麼慢慢長出來的。

## 目前結果裡，test 最好的是哪一組

目前這四組裡，test 表現最好的是 `q2-l2-e20`。它最後的 test MSE 大約是 `0.2139`，訓練過程中的最低點大約到 `0.1893`。第二好的是 `q2-l3-e20`。`q3-l2-e20` 和 `q3-l3-e20` 就差得比較明顯，test MSE 都高不少。

這一輪結果看起來很直接：模型不是越大越好。至少在這個 split 上，`q=2` 反而比 `q=3` 更穩。layers 從 2 到 3 有變化，但沒有出現「加深之後就明顯解決 test 問題」的情況。

## 從訓練過程裡會看到什麼

把 slider 往前拖時，最明顯的現象通常是 train 那邊很快就貼上去，曲面會越來越順，train MSE 也會一路下降。test 那邊就不太一樣了。前面幾步通常會先進步一段，但很快就進入比較平的平台區。下面的 loss 圖也會反映同一件事：train loss 持續往下掉，但 test MSE 並沒有跟著一直改善。

這也是這個頁面最值得看的地方。只看最後一個數字，很容易以為模型只是「好或不好」；把 slider 拖過一遍之後，會更清楚它其實有學到東西，只是後面新增的訓練步數大多是在修 train domain，不一定真的幫到 test domain。

## 為什麼這題會難 fit

問題主要出在 split 本身。train domain 在左下角，那裡看到的是 target function 靠近高值區的一小段弧面；test domain 在右上角，曲面會往下彎得更明顯。也就是說，模型在 train 上學到的是一段局部形狀，接下來卻要把這段形狀延伸到另一塊長得不太一樣的區域。

所以這題比較像 extrapolation，不是單純的 interpolation。這也解釋了為什麼 train MSE 已經很小，test 還是可能卡住。模型不是沒有學到，而是它能從 train 區域拿到的訊息本來就有限。

## 頻譜在看什麼

3D surface 看的是空間裡的形狀，Fourier spectrum 看的是這個形狀裡有哪些頻率成分。這兩個角度放在一起看會比較完整。

就這一輪結果來看，`q2-l2` 和 `q2-l3` 的頻譜明顯比較接近 target。它們抓到的主頻位置和 target 比較一致，只是整體能量還是弱一些。相對地，`q3-l2` 和 `q3-l3` 的頻譜能量小很多，代表它們雖然也學到一些低頻結構，但重建出來的內容更弱、更平，這和它們在 test MSE 上明顯落後是同一個方向。

所以頻譜圖在這裡的用途，不只是補一張漂亮的圖，而是幫忙確認 3D surface 上看到的現象：`q2-*` 這兩組不只是表面上看起來比較像，它們在主要頻率成分上也更接近 target；`q3-*` 這兩組則比較像只抓到了一個更弱、更平滑的近似面。
