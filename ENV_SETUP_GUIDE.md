# 📋 .env 完整配置指南

所有敏感資訊（tokens、密鑰、API keys）都從 `.env` 檔案讀取，安全且方便。

---

## 🚀 快速開始

### Step 1: 複製範本

```bash
cp .env.example .env
```

### Step 2: 編輯 .env 並填入你的 tokens

```bash
# 推薦編輯工具
nano .env
# 或
vim .env
# 或
code .env  # VS Code
```

### Step 3: 驗證配置已加載

```python
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv('TELEGRAM_BOT_TOKEN'))  # 應該顯示你的 token
```

---

## 📱 各服務配置詳解

### 1️⃣ **Telegram** （推薦 - 最可靠的通知方式）

#### 為什麼選 Telegram？
- ✅ 速度最快
- ✅ 支援群組和頻道
- ✅ 完全免費
- ✅ 支援 Markdown 格式
- ✅ 可靠性最高（99.9% uptime）

#### 配置步驟

**Step 1: 建立 Bot**
```
1. 在 Telegram 中搜索 @BotFather
2. 傳送命令: /newbot
3. 按提示輸入 Bot 名稱和用戶名
4. 複製返回的 Token (格式: 123456:ABCdefGHIjklmnoPQRstuvWXYZ)
```

**Step 2: 取得 Chat ID**
```
1. 在 Telegram 中搜索 @userinfobot
2. 傳送: /start
3. Bot 返回你的 User ID
   
   - 個人對話: 直接使用返回的 ID
   - 群組/頻道: 先將 bot 添加到群組 → 發送消息 → 查看日誌獲取 Chat ID
```

**Step 3: 填入 .env**
```bash
TELEGRAM_BOT_TOKEN=123456:ABCdefGHIjklmnoPQRstuvWXYZ
TELEGRAM_CHAT_ID=987654321
```

#### 測試
```python
import os
from dotenv import load_dotenv
import requests

load_dotenv()

token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

# 發送測試消息
response = requests.post(
    f'https://api.telegram.org/bot{token}/sendMessage',
    json={'chat_id': chat_id, 'text': 'Bot is working! ✅'}
)
print(response.json())
```

---

### 2️⃣ **Email** （Gmail App Password）

#### 配置步驟

**Step 1: 啟用 2FA 與生成 App Password**
```
1. 訪問 https://myaccount.google.com/
2. 左側選擇 "Security" (安全)
3. 確保已啟用 2-Step Verification
4. 在 "App passwords" 中生成新密碼
   - 選擇應用: Mail
   - 選擇設備: Windows PC (或你的平台)
5. 複製返回的 16 字元密碼 (含空格)
```

**Step 2: 填入 .env**
```bash
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx  # App password (含空格)
EMAIL_RECIPIENT=your_email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

#### 測試
```python
import smtplib
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

load_dotenv()

server = smtplib.SMTP(os.getenv('SMTP_SERVER'), int(os.getenv('SMTP_PORT')))
server.starttls()
server.login(os.getenv('EMAIL_SENDER'), os.getenv('EMAIL_PASSWORD'))

msg = MIMEText('Test email from crypto bot!')
msg['Subject'] = 'Crypto Bot Test'
msg['From'] = os.getenv('EMAIL_SENDER')
msg['To'] = os.getenv('EMAIL_RECIPIENT')

server.send_message(msg)
server.quit()
print("✅ Email sent successfully!")
```

---

### 3️⃣ **Discord** （三種方式）

#### 方式 A: Webhook（推薦用於通知）

```
1. 右擊 Discord 頻道
2. 選擇 "Edit Channel"
3. 左側選擇 "Integrations"
4. 點擊 "Webhooks" → "New Webhook"
5. 設定名稱和圖示
6. 點擊 "Copy Webhook URL"
```

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

#### 方式 B: Bot Token（推薦用於互動式 bot）

```
1. 訪問 https://discord.com/developers/applications
2. "New Application" → 輸入名稱
3. 左側選擇 "Bot" → "Add Bot"
4. 在 "TOKEN" 下點擊 "Copy" 複製 token
5. 在 "SCOPES" 選擇 bot
6. 在 "PERMISSIONS" 選擇需要的權限
7. 複製下方的邀請 URL 並訪問
```

```bash
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_CHANNEL_ID=your_channel_id_here
```

#### 方式 C: 提及角色

```
# 獲取 Role ID
1. 在 Discord 伺服器中啟用開發者模式
   Settings → Advanced → Developer Mode
2. 右擊角色 → "Copy Role ID"
```

```bash
DISCORD_ALERT_ROLE_ID=your_role_id_here
```

#### 測試 Webhook
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

data = {
    'content': '🤖 Crypto Bot is online!',
    'tts': False
}

response = requests.post(webhook_url, json=data)
print(f"Status: {response.status_code}")
```

---

### 4️⃣ **HuggingFace** （模型存儲）

#### 配置步驟

**Step 1: 生成 Token**
```
1. 訪問 https://huggingface.co/settings/tokens
2. "New token" → 輸入名稱
3. 選擇 "Write" 權限（用於上傳模型）
4. 複製 token
```

**Step 2: 創建倉庫**
```
1. 訪問 https://huggingface.co/new
2. Repository name: crypto-price-predictor-v8
3. License: MIT
4. 創建倉庫
```

**Step 3: 填入 .env**
```bash
USE_HUGGINGFACE_MODELS=true
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_REPO_ID=username/crypto-price-predictor-v8
```

#### 使用
```bash
# 上傳模型到 HF
python upload_to_hf.py

# 從 HF 下載模型（VM）
python download_from_hf.py
```

---

### 5️⃣ **Binance API** （可選 - 交易數據）

#### 配置步驟

```
1. 訪問 https://www.binance.com/en/account/api-management
2. "Create API Key"
3. 設置：
   - Restrict access to trusted IPs only
   - 只勾選需要的權限: Read, Spot Trading
4. 複製 API Key 和 Secret Key
```

```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret_key
```

---

### 6️⃣ **Groq API** （可選 - AI 信號驗證）

#### 配置步驟

```
1. 訪問 https://console.groq.com
2. 登錄 / 註冊
3. 選擇 "Keys" → "Create API Key"
4. 複製 API Key
```

```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🛡️ 安全最佳實踐

### ✅ DO (做這些)

```bash
# 使用 .env.example 作為範本
cp .env.example .env

# 驗證 .gitignore 包含 .env
grep ".env" .gitignore

# 確保 .env 權限正確
chmod 600 .env

# 定期更新和輪換 tokens
# 使用強密碼和隨機生成的 tokens
```

### ❌ DON'T (不要這樣)

```bash
# ❌ 不要硬編碼 tokens
token = "hf_xxxx"  # 危險!

# ❌ 不要提交 .env 到 Git
git add .env  # 危險!

# ❌ 不要分享 .env 檔案
email .env to someone  # 危險!

# ❌ 不要在公共地方洩露 tokens
# 例如: GitHub issue、Stack Overflow 等
```

---

## 🔍 配置驗證

### 檢查所有必要的 Tokens

```python
import os
from dotenv import load_dotenv

load_dotenv()

# 必需的配置
required = [
    'TELEGRAM_BOT_TOKEN',
    'TELEGRAM_CHAT_ID',
    'HUGGINGFACE_TOKEN',
]

for key in required:
    value = os.getenv(key)
    if value:
        print(f"✅ {key}: {value[:10]}...")
    else:
        print(f"❌ {key}: NOT SET")
```

---

## 📝 .env 格式規則

### ✅ 正確格式

```bash
# 標準格式
KEY=value

# 含空格
KEY=value with spaces

# 含特殊字符
KEY=xxxx xxxx xxxx xxxx

# 註釋
# This is a comment
KEY=value
```

### ❌ 錯誤格式

```bash
# ❌ 多餘空格
KEY = value

# ❌ 引號
KEY="value"
KEY='value'

# ❌ 特殊字符未轉義
KEY=password!@#$%
```

---

## 🐛 常見問題

### Q1: .env 無法被讀取

```python
# ❌ 錯誤: 忘記 load_dotenv()
import os
token = os.getenv('HF_TOKEN')  # None

# ✅ 正確: 先加載 .env
from dotenv import load_dotenv
import os

load_dotenv()  # 必須先執行
token = os.getenv('HF_TOKEN')  # 有值
```

### Q2: Token 過期

```
症狀: API 返回 401 Unauthorized
解決:
1. 檢查 .env 中的 token 是否正確
2. 訪問服務官網重新生成 token
3. 更新 .env 並重啟應用
```

### Q3: 特殊字符問題

```bash
# 如果 token 包含特殊字符，不需要引號
# 正確
TOKEN=abc!@#$%^&*()

# 錯誤
TOKEN="abc!@#$%^&*()"
```

---

## 📚 相關檔案

| 檔案 | 用途 |
|------|------|
| `.env` | 你的個人配置（不提交） |
| `.env.example` | 配置範本（提交到 Git） |
| `.gitignore` | 包含 `.env` 規則 |
| `upload_to_hf.py` | 自動讀取 `.env` 中的 HF_TOKEN |
| `download_from_hf.py` | 自動讀取 `.env` 中的 HF_TOKEN |
| `bot_predictor.py` | 可擴展以讀取所有 tokens |

---

## ✨ 下一步

```bash
# 1. 複製範本
cp .env.example .env

# 2. 編輯並填入 tokens
nano .env

# 3. 安裝依賴
pip install python-dotenv

# 4. 測試配置
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('✅' if os.getenv('TELEGRAM_BOT_TOKEN') else '❌')"

# 5. 開始使用
python upload_to_hf.py
python download_from_hf.py
```

---

**最後更新**: 2025-12-14

**狀態**: ✅ 生產就緒
