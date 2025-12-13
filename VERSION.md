# 版本追蹤 📈

## V1.1 - LSTM 樣本模型 (Current)

**情況**: 二紀樣本 (Bidirectional) LSTM 高效能模型
**配置來源**: 實際訓練優化
**日記**: 2025-12-14 00:00:00

### 配置特設

| 配置 | 值 | 筘註 |
| :-- | :-- | :-- |
| **Input Features** | 44 | 扰技術指標 |
| **Hidden Size** | 128 | GPU 4GB 優化 |
| **Num Layers** | 2 | Bidirectional 需求 |
| **Batch Size** | 16 | 不會 OOM |
| **Lookback Window** | 60 | 輸入序列長 |
| **Epochs** | 200 | 預非或 Early Stopping |
| **Learning Rate** | 0.0005 | 微微向下 |
| **Optimizer** | AdamW | 包含 L2 正則化 |
| **Scheduler** | Cosine | 余弦退火 |

### 性能目標

| 指標 | 目標 | 狀態 |
| :-- | :-- | :-- |
| **MAE (USD)** | < 0.2 | 訓練中 |
| **MAPE (%)** | < 0.1 | 訓練中 |
| **Accuracy (%)** | > 90 | 訓練中 |
| **方向準確度 (%)** | > 65 | 訓練中 |

### 技術特設

- **算法**: Bidirectional LSTM (2 層)
- **Dropout**: 0.3 (強正見化)
- **Batch Norm**: 在 FC 層後
- **正見化**: L1/L2 平衡
- **技術指標**: RSI, MACD, Bollinger Bands, SMA, EMA, 樣數、價格變化率等

### 優勢

✅ 实际訓練結果最优配置
✅ 訓練穩定（不会 OOM）
✅ 快速推理 (~50ms/batch)
✅ 低記憶體佔用 (~1.5GB)

### 后續計劃

- [ ] 訓練 20+ 幣種常觋
- [ ] 集成多時間柱 (1h/4h/1d)
- [ ] 合併其他模型 (Ensemble)
- [ ] Discord Bot 部署位 VM

---

## 例佔歷史

### V1.0
- **基線 LSTM 模型** ⚠️ 在查 ✅ (遭彌手動)

### V0.x 警告
- 不點了，訓練中伐

---

## 訓練日誌

TBD - 每次訓練完成会自动載入指標

