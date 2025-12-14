# 🖥️ VM 部署完整指南

在雲端 VM（AWS EC2、Azure VM、DigitalOcean 等）上部署 Crypto Bot

---

## 📋 前置準備

### 系統需求

```
OS: Linux (Ubuntu 20.04 or later)
Python: 3.8+
CPU: 2+ cores
RAM: 4GB+
磁碟: 20GB+ (含模型)
Network: 穩定連接
```

### 帳戶需求

```
✅ HuggingFace 帳戶 + HUGGINGFACE_TOKEN
✅ Telegram Bot Token + Chat ID
✅ Discord Bot Token + Channel ID (可選)
✅ Binance API Key (可選)
```

---

## 🚀 Step 1: 初始設置 (首次部署)

### 連接到 VM

```bash
# SSH 登錄
ssh -i your-key.pem user@vm-ip

# 或使用密碼
ssh user@vm-ip
```

### 更新系統

```bash
sudo apt update && sudo apt upgrade -y

# 安裝必要工具
sudo apt install -y python3 python3-pip git curl wget
```

### 建立工作目錄

```bash
# 建立專案目錄
mkdir -p ~/crypto-bot
cd ~/crypto-bot

# 克隆專案
git clone https://github.com/caizongxun/crypto-trading-bot-ml.git .

# 或如果已存在，就更新
git pull origin main
```

---

## 🐍 Step 2: Python 虛擬環境

### 建立虛擬環境

```bash
# 建立 venv
python3 -m venv venv

# 激活虛擬環境
source venv/bin/activate  # Linux/Mac
# 或 venv\\Scripts\\activate  # Windows

# 驗證
which python  # 應該在 venv 內
```

### 安裝依賴

```bash
# 升級 pip
pip install --upgrade pip

# 安裝所有依賴
pip install -r requirements.txt

# 驗證安裝
python -c "import torch; print(torch.__version__)"
```

---

## 🔑 Step 3: 配置 .env

### 複製 .env 範本

```bash
cp .env.example .env
```

### 編輯 .env

```bash
# 推薦用 nano 編輯
nano .env

# 或用 vim
vim .env

# 或用 cat + heredoc
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_REPO_ID=caizongxun/crypto-price-predictor-v8
DISCORD_BOT_TOKEN=your_discord_token (可選)
BINANCE_API_KEY=your_api_key (可選)
EOF
```

### 驗證 .env

```bash
# 檢查格式
cat .env

# 驗證能否讀取
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF_TOKEN:', os.getenv('HUGGINGFACE_TOKEN')[:20]+'...')"
```

---

## 📥 Step 4: 下載模型（推薦方案）

### 選項 A: 一次性下載所有模型

```bash
# 下載所有模型到 models/saved/
python download_from_hf.py

# 這會下載:
# - 20 個 .pth 模型檔案 (~1-2 GB)
# - bias_corrections_v8.json
# - bot_predictor.py

# 驗證下載
ls -lh models/saved/
du -sh models/saved/  # 檢查大小
```

### 選項 B: 按需下載（節省空間）

如果 VM 儲存不足，只下載需要的模型：

```python
# download_selective.py
from huggingface_hub import hf_hub_download

# 只下載 BTC 和 ETH 模型
symbols = ['BTC', 'ETH']

for symbol in symbols:
    print(f"Downloading {symbol} model...")
    hf_hub_download(
        repo_id="caizongxun/crypto-price-predictor-v8",
        filename=f"models/{symbol}_model_v8.pth",
        cache_dir="models/saved",
        force_download=False
    )
    print(f"✓ {symbol} model ready")
```

使用：
```bash
python download_selective.py
```

---

## 🎯 Step 5: 測試模型

### 快速測試

```bash
# 測試 BTC 預測
python -c "
from bot_predictor import BotPredictor
bot = BotPredictor()
print('Testing BTC prediction...')
prediction = bot.predict('BTC')
if prediction:
    print(f'✓ Prediction successful')
    print(f'  Current: ${prediction[\"current_price\"]:.2f}')
    print(f'  Predicted: ${prediction[\"corrected_price\"]:.2f}')
    print(f'  Direction: {prediction[\"direction\"]}')
else:
    print('✗ Prediction failed')
"
```

### 詳細測試

```bash
# 創建測試腳本
cat > test_bot.py << 'EOF'
from bot_predictor import BotPredictor
import time

bot = BotPredictor()
symbols = ['BTC', 'ETH', 'SOL']

for symbol in symbols:
    print(f"\nTesting {symbol}...")
    try:
        prediction = bot.predict(symbol)
        if prediction:
            print(f"  ✓ Success")
            print(f"    Current: ${prediction['current_price']:.2f}")
            print(f"    Predicted: ${prediction['corrected_price']:.2f}")
            print(f"    Confidence: {prediction['confidence']*100:.1f}%")
        else:
            print(f"  ✗ Failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    time.sleep(1)  # Rate limit
EOF

python test_bot.py
```

---

## 🔄 Step 6: 定期更新模型

### 方案 A: 每週自動下載

```bash
# 編輯 crontab
crontab -e

# 每週一凌晨 2 點執行
0 2 * * 1 cd /home/user/crypto-bot && source venv/bin/activate && python download_from_hf.py >> logs/download.log 2>&1
```

### 方案 B: 手動更新

```bash
# 進入 VM 並執行
cd ~/crypto-bot
source venv/bin/activate
python download_from_hf.py
```

---

## 🚀 Step 7: 啟動 Bot

### 啟動 Telegram Bot（範例）

```python
# bot_main.py
from bot_predictor import BotPredictor
import telebot
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

predictor = BotPredictor()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🤖 Crypto Bot 已啟動！")

@bot.message_handler(commands=['predict'])
def predict_price(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "用法: /predict BTC")
        return
    
    symbol = args[1].upper()
    prediction = predictor.predict(symbol)
    
    if prediction:
        text = f"""
📊 {symbol} 預測

💰 當前: ${prediction['current_price']:.2f}
🎯 預測: ${prediction['corrected_price']:.2f}
📈 方向: {prediction['direction']}
🎲 信心: {prediction['confidence']*100:.1f}%
        """
        bot.reply_to(message, text)
    else:
        bot.reply_to(message, f"❌ {symbol} 預測失敗")

if __name__ == '__main__':
    print("Bot 啟動中...")
    bot.infinity_polling()
EOF

# 執行
python bot_main.py
```

### 後台運行

```bash
# 方案 1: 使用 screen
screen -S crypto-bot
python bot_main.py
# 按 Ctrl+A 再按 D 進入背景

# 查看
screen -ls

# 重新連接
screen -r crypto-bot
```

```bash
# 方案 2: 使用 nohup
nohup python bot_main.py > logs/bot.log 2>&1 &

# 查看日誌
tail -f logs/bot.log
```

```bash
# 方案 3: 使用 systemd (推薦)
sudo cat > /etc/systemd/system/crypto-bot.service << 'EOF'
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/crypto-bot
ExecStart=/home/ubuntu/crypto-bot/venv/bin/python bot_main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl start crypto-bot
sudo systemctl enable crypto-bot

# 查看狀態
sudo systemctl status crypto-bot

# 查看日誌
sudo journalctl -u crypto-bot -f
```

---

## 📊 磁碟空間管理

### 檢查磁碟使用

```bash
# 總體狀況
df -h

# 模型大小
du -sh models/

# 按目錄排序
du -sh *| sort -hr
```

### 模型大小估計

```
BTC/ETH/SOL 模型: ~80-120 MB each
20 個模型: ~1.5-2 GB
Log 檔案: 根據運行時間 (可清理)
Cache: ~/.cache/huggingface/ (可清理)
```

### 清理空間

```bash
# 清理舊日誌
rm -f logs/*.log

# 清理 HuggingFace 緩存
rm -rf ~/.cache/huggingface/

# 清理 Python 緩存
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name '*.pyc' -delete
```

---

## 🔍 監控和調試

### 查看進程

```bash
# 查看 Python 進程
ps aux | grep python

# 監控資源使用
top
htop  # 如果已安裝
```

### 查看日誌

```bash
# 實時日誌
tail -f logs/bot.log

# 最後 100 行
tail -100 logs/bot.log

# 搜索錯誤
grep ERROR logs/bot.log
```

### 重啟 Bot

```bash
# Systemd
sudo systemctl restart crypto-bot

# Screen
screen -r crypto-bot
Ctrl+C  # 停止
exit    # 退出 screen
screen -S crypto-bot  # 重新啟動
```

---

## 🚨 故障排除

### 模型下載失敗

```bash
# 檢查網路
ping huggingface.co

# 檢查 token
grep HUGGINGFACE_TOKEN .env

# 重新下載
rm -rf models/saved/*
python download_from_hf.py
```

### Bot 無法連接 Telegram

```bash
# 檢查 token
grep TELEGRAM_BOT_TOKEN .env

# 測試 token
curl https://api.telegram.org/botTOKEN/getMe
```

### 記憶體不足

```bash
# 檢查內存
free -h

# 啟用交換空間（如果沒有）
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📝 檢查清單

部署完成後確認：

- [ ] Python 3.8+ 已安裝
- [ ] 虛擬環境已建立和激活
- [ ] 依賴已安裝 (`pip list`)
- [ ] .env 已配置（token 有效）
- [ ] 模型已下載 (`ls models/saved/`)
- [ ] 測試預測成功
- [ ] Bot 正在後台運行
- [ ] 日誌可正常查看
- [ ] 磁碟空間充足 (> 5 GB)
- [ ] 網路連接穩定

---

## 🎯 生產環境最佳實踐

```bash
# 1. 定期備份
sudo crontab -e
# 每天備份 .env
0 3 * * * tar -czf ~/backup/crypto-bot-$(date +\%Y\%m\%d).tar.gz ~/crypto-bot

# 2. 監控 Bot 狀態
# 在 cron 中定期檢查進程
*/5 * * * * ps aux | grep -q "python bot_main.py" || systemctl restart crypto-bot

# 3. 定期日誌輪轉
sudo apt install logrotate
sudo cat > /etc/logrotate.d/crypto-bot << 'EOF'
/home/ubuntu/crypto-bot/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
}
EOF
```

---

**最後更新**: 2025-12-14

**狀態**: ✅ 生產就緒
