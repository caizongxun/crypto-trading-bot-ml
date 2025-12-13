# 加密貨幣價格預測系統 🚀

本地訓練模型 → Discord Bot VM 實時推理

## 📋 項目結構

```
crypto-trading-bot-ml/
├── training/              # 本地訓練腳本（可久、模型可大）
│   ├── train_lstm_v1.py   # LSTM 訓練（效果好，MAE < 0.2）
│   ├── data_fetcher.py    # 數據獲取
│   ├── config.yaml        # 訓練配置
│   └── requirements.txt
├── models/                # 訓練好的模型
│   └── saved/             # 模型檔案存放
├── discord_bot/           # Discord Bot 部署代碼
│   ├── bot.py             # Bot 主程序
│   ├── predictor.py       # 推理引擎
│   └── handlers/
├── .gitignore             # 排除 .env 和本地資源
├── README.md
└── VERSION.md             # 版本追蹤

## 🏃 工作流程

### 1️⃣ 本地訓練（可以久）
```bash
cd training
python train_lstm_v1.py --symbol SOL --epochs 200
```
輸出: `models/saved/SOL_model.pth` + 訓練指標

### 2️⃣ 上傳到 GitHub
- 模型自動推送到 `models/saved/`
- 訓練指標記錄在 `training/results/`
- 每個版本記錄 MAE、MAPE 等

### 3️⃣ VM 拉取 & 推理
```bash
git pull  # 拉取最新模型
python discord_bot/bot.py  # 啟動 Bot
```
Bot 自動讀取本地模型，推理價格，推送 Discord 通知

## 📊 版本追蹤

每次訓練完成自動提交：
```
1-[功能]: 訓練 [幣種]，MAE=X.XX, MAPE=Y.YY%, Accuracy=Z.Z%
2-[重大突破]: 模型升級，5 種幣種，平均 MAE < 1.0
```

## ⚙️ 配置

- **本地**: `.env` （已有，不上傳）
  - `BINANCE_API_KEY`
  - `BINANCE_SECRET`
  - `DISCORD_TOKEN`
  - `HF_TOKEN`

- **VM**: git pull 後自動用本地 `.env`

## 🚀 快速開始

1. **訓練 (Local)**
   ```bash
   git clone https://github.com/caizongxun/crypto-trading-bot-ml.git
   cd training
   python train_lstm_v1.py --symbol SOL --epochs 200
   ```

2. **推送到 GitHub**
   - 訓練腳本自動 `git add` + `commit` + `push`
   - 版本號自動遞增

3. **部署到 VM**
   ```bash
   git pull
   python discord_bot/bot.py
   ```

## 📈 預期效果

- **MAE**: < 2 USD（好版本）
- **MAPE**: < 2%
- **方向準確度**: > 70%
- **推理延遲**: < 500ms

## 🔄 文件名稱規則

**不要自己創建版本**，除非明確說版本更新，否則都是 `v1`

- `train_lstm_v1.py` ✅ 保持 v1
- `SOL_model.pth` ✅ 保持無版本號
- `config_v2.yaml` ❌ 不要自己創建 v2

---

**版本**: V1.0 | **狀態**: 開發中 🛠️
