# 🚀 快速開始

從 0 到有的最快方案！

---

## 🌟 選項 1：本機開發（完整版）

最推薦，可以訓練、檢測、上傳模型

### Step 1: 克隆並安裝

```bash
# 克隆仓库
git clone https://github.com/caizongxun/crypto-trading-bot-ml.git
cd crypto-trading-bot-ml

# 建立 Python 虫简環境
Python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### Step 2: 配置 .env

```bash
# 複製範本
cp .env.example .env

# 編輯 .env
nano .env  # 或 vim / VS Code

# 填入你的 tokens
# - TELEGRAM_BOT_TOKEN
# - TELEGRAM_CHAT_ID
# - HUGGINGFACE_TOKEN
# - 其他可選配置
```

詳解：親自閱讀 `ENV_SETUP_GUIDE.md` 了解如何獲取各項 tokens

### Step 3: 訓練模型

```bash
# 訓練所有 20 個幣種
Python train_v8_models.py

# 或訓練單一幣種
Python train_v8_single.py --symbol BTC
```

### Step 4: 檢測偏差並校正

```bash
# 自動檢測所有模型的偏差
Python detect_all_shifts.py

# 結果: 算出你所有模型的偏差值
# 自動保存到 models/bias_corrections_v8.json
```

### Step 5: 可視化校正結果

```bash
# 生成校正前後對比圖表
Python visualize_all_v8_corrected.py

# 結果存在 output/文件夼
```

### Step 6: 上傳到 HuggingFace

```bash
# 直接上傳整個 models/saved/ 資料夼
# 自動讀取 .env 中的 HF_TOKEN
Python upload_to_hf.py

# 結果: 驗證成功
# 數據儲存位置: https://huggingface.co/username/crypto-price-predictor-v8
```

---

## ☄️ 選項 2：VM 部署（最小化）

從 HuggingFace 下載模型，直接集成到 Discord Bot 或 Telegram Bot

### Step 1: 取得你的 .env

```bash
# 方式 1: 從本機複製
# 將本機的 .env 複製到 VM

# 方式 2: 手動檔案
cp .env.example .env
nano .env  # 填入 HF_TOKEN 等扥訊
```

### Step 2: 安裝依賴

```bash
# VM 繁神設置
cd /home/username/crypto-trading-bot-ml

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: 從 HuggingFace 下載模型

```bash
# 自動讀取 .env 中的 HF_TOKEN
Python download_from_hf.py

# 結果: 下載
# - models/saved/*.pth (所有 20 個模型)
# - bias_corrections_v8.json
# - bot_predictor.py
```

### Step 4: 測試預測

```bash
# 測試模型是否正常工作
Python -c "from bot_predictor import BotPredictor; bot = BotPredictor(); print(bot.predict('BTC'))"
```

### Step 5: 集成到 Discord/Telegram Bot

```python
# 在你的 Bot 代碼中

from bot_predictor import BotPredictor
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化
bot = BotPredictor()

# 取得預測
def get_crypto_signal(symbol):
    prediction = bot.predict(symbol)
    if prediction:
        return f"""
📊 {symbol}
💵 當前: ${prediction['current_price']:.2f}
🎯 預測: ${prediction['corrected_price']:.2f}
↗️ 方向: {prediction['direction']}
🌟 信心: {prediction['confidence']*100:.1f}%
        """
    return f"❌ {symbol} 預測失敗"

# 在 Discord 或 Telegram 中使用
print(get_crypto_signal('BTC'))
print(get_crypto_signal('ETH'))
```

---

## 💪 後續筆記

### 訓練中的不同檔案

| 檔案 | 用途 |
|--------|------|
| `train_v8_models.py` | 訓練所有 20 個幣種 |
| `train_v8_single.py` | 訓練單一幣種 |
| `train_v8_enhanced.py` | 增強訓練（有驗證和早停） |
| `diagnose_shift.py` | 診斷單一模型偏差 |
| `detect_all_shifts.py` | 掃描所有偏差 |
| `visualize_all_v8_corrected.py` | 可視化校正結果 |
| `bot_predictor.py` | Bot 預測模組 |
| `upload_to_hf.py` | 上傳到 HuggingFace |
| `download_from_hf.py` | 從 HuggingFace 下載 |

### 下次墨水

```bash
# 定時更新模型
crontab -e
# 添加: 0 0 * * * cd /path && python train_v8_models.py

# 定時推送訓練信號
# 使用 APScheduler 或 Celery
```

---

## ⚠️ 常見問題

### Q: 模型訓練失敗

```bash
# 倉庫不存在
python train_v8_models.py
# 驗證 models/saved/ 目錄是否存在
mkdir -p models/saved
```

### Q: 上傳失敗

```bash
# Token 有效
# 確保 .env 正確斷後 HF_TOKEN
grep HF_TOKEN .env

# 小心別注釋或餘佐
 nano .env  # 確保沒有余餘斷
```

### Q: 預測不準確

```bash
# 梨应是二次訓練 / 訃參數
python train_v8_enhanced.py

# 梨經過 bias 校正
detect_all_shifts.py
```

---

## 🏁 完綄!

成功完成了以下步驟:

- ✅ 訓練模型
- ✅ 校正偏差
- ✅ 上傳到 HuggingFace
- ✅ 集成到 Bot
- ✅ 實配到 VM
- ✅ 開始發送交易信號

梨歩篶先感受！🌟

---

## 📄 詳解文檔

- `README.md` - 主文檔
- `ENV_SETUP_GUIDE.md` - .env 配置詳解
- `PROJECT_STRUCTURE.md` - 专案結構
- `DEPLOYMENT_GUIDE.md` - VM 部署指南
- `QUICK_START.md` - 本檔案（当前阋）

**最後更新**: 2025-12-14
