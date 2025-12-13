# 可視化指南 📊

## 概述

`visualize_results.py` 工具為你生成 **6 個綜合圖表**，清楚展示模型預測準確度與實際價格線的對齐情況。

## 快速開始

### 基本用法

```bash
# 訓練完成後，立即可視化結果
python training/visualize_results.py --symbol SOL
```

### 進階用法

```bash
# 自訂數據點數量
python training/visualize_results.py --symbol SOL --limit 500

# 自訂模型和結果路徑
python training/visualize_results.py --symbol SOL \
  --model models/saved/SOL_model.pth \
  --results results/SOL_results.json

# 其他幣種
python training/visualize_results.py --symbol BTC
python training/visualize_results.py --symbol ETH
```

## 圖表說明

### 1️⃣ 價格預測對比 (左上)

**功能**: 展示實際價格線與預測價格線的重疊情況

- **藍線**: 實際價格（訓練中得到的真實標籤）
- **橙色虛線**: 模型預測價格
- **灰色陰影**: 兩者之間的差距

**如何解讀**:
- ✅ 藍線與橙線越接近 → 預測越準確
- ❌ 如果線條差距太大 → 模型需要優化

**目標**: 線條應該 **99%+ 重疊**

---

### 2️⃣ 誤差分布直方圖 (右上)

**功能**: 展示預測誤差的統計分布

- **橫軸**: 絕對誤差大小 (USD)
- **縱軸**: 誤差發生的頻率
- **紅色虛線**: 平均誤差 (MAE)

**如何解讀**:
- ✅ 分布集中在 0 附近 → 大多數預測都很準確
- ✅ 呈現正態分布 → 模型表現穩定
- ❌ 分布分散 → 模型預測不穩定

**目標**: MAE < $0.2 USD

---

### 3️⃣ 散點圖 (左中)

**功能**: 逐點比較實際價格 vs 預測價格

- **横軸**: 實際價格
- **縱軸**: 預測價格
- **黑色虛線**: 完美預測線 (y=x)
- **紅點**: 各個預測點

**如何解讀**:
- ✅ 點集中在虛線上 → 預測非常準確
- ✅ R² 分數接近 1.0 → 模型解釋了 99%+ 的價格變化
- ❌ 點分散 → 預測精度低

**目標**: R² > 0.90（最好 > 0.95）

---

### 4️⃣ 誤差時間序列 (右中)

**功能**: 查看誤差如何隨時間變化

- **綠色柱**: 預測值高於實際值
- **紅色柱**: 預測值低於實際值
- **黑色水平線**: 誤差 = 0 的參考線

**如何解讀**:
- ✅ 柱子長度接近 0 → 預測誤差小
- ✅ 綠紅柱交替 → 模型誤差不偏向
- ❌ 長期偏向一邊 → 模型存在系統性偏差

**目標**: 誤差應該在 ±0.5 USD 以內

---

### 5️⃣ 性能指標表 (左下)

**功能**: 關鍵指標一覽

| 指標 | 說明 | 目標 |
| :-- | :-- | :-- |
| **MAE** | 平均絕對誤差 | < $0.2 |
| **MAPE** | 平均絕對百分比誤差 | < 0.1% |
| **RMSE** | 均方根誤差（懲罰大誤差） | < $0.3 |
| **R²** | 決定係數 | > 0.90 |
| **方向準確度** | 預測上升/下降的準確率 | > 65% |
| **測試樣本數** | 用於評估的數據點 | 越多越好 |
| **訓練狀態** | 當前模型版本 | v1.1 |

✅ 綠色勾號表示指標達標
❌ 紅色 X 表示指標未達標

---

### 6️⃣ 價格變化方向對比 (右下)

**功能**: 比較實際和預測的上升/下降方向

- **藍色柱**: 實際方向（1=上升, 0=下降）
- **橙色柱**: 預測方向（1=上升, 0=下降）

**如何解讀**:
- ✅ 柱子大多重疊 → 方向預測準確
- ❌ 柱子經常不重疊 → 需要優化方向預測

**目標**: 方向準確度 > 70%

---

## 完整示例

### 訓練後立即可視化

```bash
# 1. 訓練模型
python training/train_lstm_v1.py --symbol SOL --epochs 200

# 2. 模型訓練完成後，拉取最新代碼
git pull origin main

# 3. 生成可視化
python training/visualize_results.py --symbol SOL --limit 300

# 4. 會自動生成並保存圖表到：
# results/visualizations/SOL_predictions_20251214_120000.png
```

### 評估多個幣種

```bash
for symbol in SOL BTC ETH DOGE; do
    echo "\n=== Visualizing $symbol ==="
    python training/visualize_results.py --symbol $symbol --limit 200
done
```

## 如何解讀整體結果

### ✅ 模型表現好的跡象

- ✅ 價格線基本重疊（差距 < 5%）
- ✅ 誤差分布集中在 0 附近
- ✅ 散點緊聚在完美預測線周圍
- ✅ MAE < $0.2, MAPE < 0.1%
- ✅ R² > 0.90
- ✅ 方向準確度 > 70%
- ✅ 誤差時間序列中綠紅柱均衡分布

### ❌ 模型需要優化的跡象

- ❌ 預測線與實際線差距大
- ❌ 誤差分布寬（MAE > $0.5）
- ❌ 散點分散
- ❌ R² < 0.80
- ❌ 方向準確度 < 60%
- ❌ 誤差持續偏向一側

## 優化建議

如果模型表現不理想，嘗試以下調整：

### 1. 提高特徵數量
```yaml
# 修改 training/config.yaml
model:
  hidden_size: 256  # 從 128 提高到 256
  num_layers: 3     # 從 2 增加到 3
```

### 2. 調整訓練參數
```yaml
training:
  batch_size: 8        # 從 16 降低到 8（更細粒度）
  learning_rate: 0.001 # 從 0.0005 提高
  epochs: 300          # 增加訓練輪數
```

### 3. 增加訓練數據
```bash
# 增加載入的歷史數據
python training/train_lstm_v1.py --symbol SOL --epochs 200
# 修改 training/config.yaml 的 limit: 10000
```

### 4. 嘗試其他符號
```bash
# 有些幣種可能更容易預測
python training/visualize_results.py --symbol BTC  # 波動性低
python training/visualize_results.py --symbol DOGE # 波動性高
```

## 保存和分享結果

### 位置

```
results/visualizations/
├── SOL_predictions_20251214_120000.png
├── BTC_predictions_20251214_130000.png
└── ...
```

### 批量生成所有幣種的圖表

```bash
for symbol in SOL BTC ETH DOGE XRP ADA; do
    python training/visualize_results.py --symbol $symbol --limit 300
    # 圖表自動保存到 results/visualizations/
done

# 合併所有 PNG（可選，需要 ImageMagick）
convert results/visualizations/*.png merged_results.pdf
```

## 常見問題

### Q: 圖表中沒有出現任何圖案
A: 確保模型已訓練完成，`models/saved/{symbol}_model.pth` 存在

### Q: 所有指標都紅 X
A: 模型可能未收斂，嘗試增加訓練輪數或調整超參數

### Q: 價格線雜亂無序
A: 可能是數據問題，檢查 Binance API 連接和數據質量

### Q: MAE 很高但 R² 不錯
A: 正常現象，可能是幣種波動大。考慮用百分比誤差 (MAPE) 評估

## 技術細節

### 使用的指標

- **MAE** (Mean Absolute Error): 平均絕對誤差，對異常值不敏感
- **MAPE** (Mean Absolute Percentage Error): 百分比誤差，適合不同規模資料
- **RMSE** (Root Mean Square Error): 均方根誤差，重視大誤差
- **R²**: 決定係數，模型解釋力度
- **方向準確度**: 正確預測上升/下降的百分比

### 圖表生成流程

1. 加載已訓練的模型
2. 從 Binance 獲取最新 K 線數據
3. 特徵提取與正規化
4. 模型推理
5. 計算評估指標
6. 生成 6 個子圖
7. 保存到 `results/visualizations/`

---

**祝訓練順利！** 🚀

如有問題，歡迎提 Issue 或 Discussion！
