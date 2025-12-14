# .env Setup Guide 🔐

所有敏感資訊（tokens）都從 `.env` 檔案讀取，安全且方便。

---

## 📋 快速設置

### Step 1: 複製範本

```bash
cp .env.example .env
```

### Step 2: 填入你的 Tokens

編輯 `.env` 檔案，填入以下資訊：

```bash
# HuggingFace Token (用於 upload_to_hf.py 和 download_from_hf.py)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Discord Token (用於 Discord Bot)
DISCORD_TOKEN=your_discord_bot_token

# 其他可選 Tokens
BINANCE_API_KEY=xxx
BINANCE_SECRET_KEY=xxx
GROQ_API_KEY=xxx
```

### Step 3: 確認 .env 在 .gitignore

確保 `.env` **不會被上傳到 GitHub**：

```bash
# 檢查 .gitignore
grep ".env" .gitignore

# 如果沒有，手動添加
echo ".env" >> .gitignore
```

---

## 🎯 Token 來源

### HuggingFace Token

1. 訪問：https://huggingface.co/settings/tokens
2. 點擊 "New token"
3. 選擇 "Write" 權限（用於上傳模型）
4. 複製 token，貼到 `.env`

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxx
```

### Discord Bot Token

1. 訪問：https://discord.com/developers/applications
2. 創建 "New Application"
3. 左側選擇 "Bot"
4. 點擊 "Reset Token" 並複製
5. 貼到 `.env`

```bash
DISCORD_TOKEN=your_token_here
```

### Binance API Keys (可選)

1. 訪問：https://www.binance.com/en/account/api-management
2. 創建 "New key"
3. 選擇權限：`Read` / `Trade` (根據需要)
4. 複製 API Key 和 Secret Key

```bash
BINANCE_API_KEY=xxx
BINANCE_SECRET_KEY=xxx
```

### Groq API Key (可選)

1. 訪問：https://console.groq.com/keys
2. 複製 API Key

```bash
GROQ_API_KEY=gsk_xxx
```

---

## 🚀 使用示例

### 上傳到 HuggingFace

```bash
# .env 中必須有 HF_TOKEN
python upload_to_hf.py
```

### 從 HuggingFace 下載（VM）

```bash
# .env 中可以有 HF_TOKEN（公開 repo 不需要）
python download_from_hf.py
```

### 在 Python 代碼中使用

```python
import os
from dotenv import load_dotenv

# 加載 .env
load_dotenv()

# 讀取 tokens
hf_token = os.getenv('HF_TOKEN')
discord_token = os.getenv('DISCORD_TOKEN')
binance_key = os.getenv('BINANCE_API_KEY')

print(f"HF Token: {hf_token[:10]}...")
print(f"Discord Token: {discord_token[:10]}...")
```

---

## ⚠️ 安全提示

✅ **DO:**
- 使用 `.env.example` 作為範本
- 保持 `.env` 在 `.gitignore` 中
- 定期更換敏感 token
- 為不同環境使用不同 token
- 設置 token 有效期

❌ **DON'T:**
- 提交 `.env` 到 Git
- 在代碼中硬編碼 token
- 共享你的 `.env` 檔案
- 在公開地方洩露 token
- 使用過期或不安全的 token

---

## 🔧 故障排除

### 錯誤：`.env` 未被讀取

```bash
# 確保 python-dotenv 已安裝
pip install python-dotenv

# 確保在代碼最開始調用
from dotenv import load_dotenv
load_dotenv()  # 必須在所有 import 之前
```

### 錯誤：Token 無效

1. 檢查 `.env` 格式（無引號）
2. 檢查 token 是否過期
3. 重新生成新 token

### 錯誤：權限不足

1. 驗證 token 有正確的權限
2. HuggingFace：需要 "Write" 權限
3. Discord：需要適當的 scopes

---

## 📝 .env 格式規則

```bash
# 正確格式（無引號、無空格）
HF_TOKEN=hf_xxx
DISCORD_TOKEN=xyz

# 錯誤格式（避免以下）
HF_TOKEN = hf_xxx  # ❌ 有空格
HF_TOKEN="hf_xxx"  # ❌ 有引號
HF_TOKEN='hf_xxx'  # ❌ 有引號
```

---

## ✅ 驗證設置

```python
import os
from dotenv import load_dotenv

load_dotenv()

# 檢查所有必要的 tokens
required_tokens = ['HF_TOKEN', 'DISCORD_TOKEN']

for token_name in required_tokens:
    token_value = os.getenv(token_name)
    if token_value:
        print(f"✓ {token_name} loaded")
    else:
        print(f"✗ {token_name} NOT found")
```

---

## 📚 相關檔案

- `.env` - 你的個人配置（不要提交）
- `.env.example` - 範本檔案（提交到 Git）
- `.gitignore` - Git 忽略規則
- `upload_to_hf.py` - 自動讀取 HF_TOKEN
- `download_from_hf.py` - 自動讀取 HF_TOKEN
- `bot_predictor.py` - 可以擴展以讀取所有 tokens

---

**最後更新**: 2025-12-14
