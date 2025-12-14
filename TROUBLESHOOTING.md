# 🔧 Troubleshooting Guide

常見問題解決方案

---

## 🔴 問題 1: HF_TOKEN 讀不到

### 症狀
```
2025-12-14 14:40:14,957 - ERROR - ✗ HF_TOKEN not found in .env file
ERROR -    Add HF_TOKEN=your_token to your .env file
```

### 原因
1. `.env` 檔案不在搜尋路徑中
2. `.env` 檔案存在但變數名稱拼寫錯誤
3. PyCharm 或 IDE 的虛擬環境配置問題

### 解決方案

**Step 1: 確認 .env 位置**

```bash
# 在你執行指令的目錄下，檢查 .env 是否存在
ls -la .env

# 或者在 PowerShell (Windows)
dir /a:h .env  # 檢查隱藏檔
```

**Step 2: 確認 .env 的位置應該在**

```
crypto-trading-bot-ml/  ← 專案根目錄
├── .env                ← 應該在這裡
├── .env.example
├── upload_to_hf.py
├── download_from_hf.py
├── models/
│   └── saved/
└── ...
```

**Step 3: 檢查 .env 的內容**

```bash
# 檢查內容是否正確
cat .env  # Linux/Mac
type .env  # Windows cmd

# 應該看到類似這樣
# TELEGRAM_BOT_TOKEN=...
# HF_TOKEN=hf_xxxxxxxxxxxxxxx
# ...
```

**Step 4: 確認 .env 格式**

```bash
# ✅ 正確格式
HF_TOKEN=hf_xxxxxxxxxxxxx

# ❌ 錯誤格式 (帶空格)
HF_TOKEN = hf_xxxxxxxxxxxxx

# ❌ 錯誤格式 (帶引號)
HF_TOKEN="hf_xxxxxxxxxxxxx"
```

**Step 5: 從 PyCharm 執行時的特殊處理**

PyCharm 有時會改變工作目錄。解決方法：

```bash
# 方法 1: 在 PyCharm 的 Terminal 中執行
# Terminal → 新 Terminal → 輸入指令
python upload_to_hf.py

# 方法 2: 在 PyCharm 中設定工作目錄
# Edit Configurations → Working directory → 選擇專案根目錄

# 方法 3: 指定 .env 路徑
# 編輯代碼中的 find_env_file() 函數，添加絕對路徑
```

**Step 6: 驗證 HF_TOKEN 已加載**

```python
# 執行這個測試
from dotenv import load_dotenv
import os

# 自動搜尋 .env
load_dotenv()

token = os.getenv('HF_TOKEN')
if token:
    print(f"✅ HF_TOKEN found: {token[:20]}...")
else:
    print("❌ HF_TOKEN not found")
    print(f"Current directory: {os.getcwd()}")
    print(f".env exists: {os.path.exists('.env')}")
```

---

## 🔴 問題 2: 在 PyCharm 中無法找到 .env

### 症狀
```
找不到 .env 檔案
工作目錄不正確
```

### 解決方案

**使用 Terminal (推薦)**

```bash
# 打開 PyCharm Terminal
Alt + F12  # 或 View → Tool Windows → Terminal

# 確認當前目錄
pwd  # Linux/Mac
cd   # Windows

# 進入專案根目錄
cd /path/to/crypto-trading-bot-ml

# 確認 .env 存在
ls .env  # Linux/Mac
dir .env  # Windows

# 執行指令
python upload_to_hf.py
```

**配置 PyCharm Run Configuration**

```
1. PyCharm Menu → Run → Edit Configurations
2. 選擇或新建 Python configuration
3. Script path: upload_to_hf.py
4. Working directory: /path/to/crypto-trading-bot-ml (選擇專案根目錄)
5. Environment variables: 可選，留空讓程式自動搜尋
6. Apply → OK
7. Run
```

**檢查虛擬環境**

```bash
# 確認虛擬環境已激活
which python  # Linux/Mac (應該指向 venv 目錄)
where python  # Windows

# 如果沒有激活，執行
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# 確認
which python  # 或 where python
```

---

## 🔴 問題 3: Token 格式錯誤

### 症狀
```
ERROR - ✗ HF_TOKEN not found in .env file
# 或
401 Unauthorized (HF API 返回)
```

### 原因
- Token 被引號包圍
- Token 有前後空格
- Token 已過期

### 解決方案

**檢查 .env 格式**

```bash
# ✅ 正確
HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456

# ❌ 錯誤 1: 有引號
HF_TOKEN="hf_abcdefghijklmnopqrstuvwxyz123456"
HF_TOKEN='hf_abcdefghijklmnopqrstuvwxyz123456'

# ❌ 錯誤 2: 有前後空格
HF_TOKEN= hf_abcdefghijklmnopqrstuvwxyz123456 
HF_TOKEN =hf_abcdefghijklmnopqrstuvwxyz123456

# ❌ 錯誤 3: 包含評論
HF_TOKEN=hf_xxx # my token
```

**驗證 Token 有效性**

```python
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HF_TOKEN')

if not token:
    print("❌ Token not found")
else:
    # 測試 token
    api = HfApi()
    try:
        # 嘗試列出用戶信息
        info = api.whoami(token=token)
        print(f"✅ Token valid for user: {info['name']}")
    except Exception as e:
        print(f"❌ Token invalid: {e}")
```

**重新生成 Token**

```
1. 訪問 https://huggingface.co/settings/tokens
2. "New token" → 選擇 "Write" 權限
3. 複製新 Token
4. 更新 .env: HF_TOKEN=hf_xxx
5. 重試
```

---

## 🔴 問題 4: 模型目錄不存在

### 症狀
```
✗ Model directory not found: models/saved
```

### 解決方案

```bash
# 建立目錄
mkdir -p models/saved

# 確認目錄結構
tree models/  # 或 ls -R models/

# 應該看到
# models/
# └── saved/
#     ├── BTC_model_v8.pth
#     ├── ETH_model_v8.pth
#     └── ...
```

---

## 🔴 問題 5: 執行 upload_to_hf.py 失敗

### 症狀
```
✗ Upload failed: ...
```

### 常見原因與解決方案

**原因 1: 模型檔案太大或網路不穩定**

```bash
# 檢查模型檔案大小
ls -lh models/saved/*.pth

# 如果超過 5GB，可能需要
# 1. 分批上傳
# 2. 改用 Git LFS
# 3. 檢查網路連接
```

**原因 2: HF 倉庫不存在或無權限**

```bash
# 確認倉庫存在
# 訪問 https://huggingface.co/username/crypto-price-predictor-v8

# 確認 token 有 Write 權限
# https://huggingface.co/settings/tokens
```

**原因 3: 超過 API 速率限制**

```python
# 解決方案: 稍候後重試
import time

# 等待 5 分鐘
time.sleep(300)

# 重新執行
python upload_to_hf.py
```

---

## 🔴 問題 6: download_from_hf.py 下載失敗

### 症狀
```
✗ Error listing repository
```

### 解決方案

**檢查網路連接**

```bash
# 測試 HuggingFace 連接
python -c "import requests; print(requests.get('https://huggingface.co').status_code)"
```

**確認倉庫公開**

```bash
# 倉庫必須是公開的才能下載
# 訪問 https://huggingface.co/caizongxun/crypto-price-predictor-v8
# 檢查 "Private" 設定
```

**查看詳細錯誤**

```python
import logging

# 啟用詳細日誌
logging.basicConfig(level=logging.DEBUG)

# 然後執行下載
python download_from_hf.py
```

---

## 🟡 提示：日誌檢查

所有腳本都會輸出詳細日誌。查看日誌可以幫助診斷問題：

```bash
# 查看完整輸出（包括 DEBUG 信息）
python upload_to_hf.py 2>&1 | tee upload.log

# 檢查日誌檔案
cat upload.log

# 搜尋錯誤
grep ERROR upload.log
```

---

## 📋 快速檢查清單

執行指令前，確認以下項目：

- [ ] `.env` 存在於專案根目錄
- [ ] `.env` 中的 `HF_TOKEN` 沒有引號或前後空格
- [ ] `HF_TOKEN` 有效且未過期 (https://huggingface.co/settings/tokens)
- [ ] HF 倉庫存在且是公開的
- [ ] `models/saved/` 目錄存在且包含 `.pth` 檔案 (上傳時)
- [ ] 網路連接正常
- [ ] Python 版本 ≥ 3.8
- [ ] 虛擬環境已激活
- [ ] 依賴已安裝: `pip install -r requirements.txt`

---

## 🆘 仍然無法解決？

### 蒐集診斷信息

```bash
# 1. 列出 Python 版本
python --version

# 2. 列出虛擬環境狀態
which python  # 或 where python

# 3. 列出當前目錄
pwd  # 或 cd

# 4. 列出 .env 內容 (隱藏敏感信息)
grep -v TOKEN .env  # 或 findstr /v TOKEN .env (Windows)

# 5. 執行診斷腳本
python -c "
import os; from pathlib import Path; from dotenv import load_dotenv
load_dotenv()
print(f'Current: {os.getcwd()}')
print(f'.env exists: {Path.cwd() / \".\".env}.exists()}')
print(f'HF_TOKEN set: {bool(os.getenv(\"HF_TOKEN\"))}')
print(f'models/saved exists: {(Path.cwd() / \"models\" / \"saved\").exists()}')
"
```

### 向社群報告

在 GitHub Issues 中報告時，請提供：

1. 完整的錯誤日誌
2. 診斷信息
3. 你的系統資訊 (OS, Python 版本)
4. 已嘗試的解決方案

---

**最後更新**: 2025-12-14

**相關文檔**:
- `ENV_SETUP_GUIDE.md` - .env 配置
- `QUICK_START.md` - 快速開始
- `README.md` - 主文檔
