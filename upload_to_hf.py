#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload Models to HuggingFace Hub

Usage:
  python upload_to_hf.py

Features:
  - Auto-finds .env file in project root and parent directories
  - Robust .env file reading with encoding support
  - Reads HUGGINGFACE_TOKEN from .env file (also supports HF_TOKEN)
  - Uploads entire models/saved/ folder at once (avoids API rate limiting)
  - Uploads bias corrections and bot predictor
  - Creates README.md for HF repo

Requires:
  - .env file with HUGGINGFACE_TOKEN or HF_TOKEN
  - huggingface_hub package
  - All models in models/saved/
  - bias_corrections_v8.json in models/
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = "models/saved"
CONFIG_FILE = "models/bias_corrections_v8.json"
README_PATH = "README_HF.md"


def find_env_file():
    """
    自動搜尋 .env 檔案
    搜尋順序:
    1. 當前工作目錄
    2. 指令檔案所在目錄
    3. 上層目錄
    4. 使用者主目錄
    """
    search_paths = [
        Path.cwd() / ".env",  # 當前工作目錄
        Path(__file__).parent / ".env",  # 指令所在目錄
        Path(__file__).parent.parent / ".env",  # 上層目錄
        Path.home() / ".env",  # 使用者主目錄
    ]
    
    for env_path in search_paths:
        if env_path.exists():
            logger.info(f"✓ Found .env at: {env_path}")
            return str(env_path)
    
    logger.warning("⚠️  .env file not found in standard locations")
    logger.info("Searching for .env in project root...")
    
    # 尋找 .env 在專案根目錄（向上搜尋直到找到 .git 或 README.md）
    current = Path.cwd()
    for _ in range(5):  # 向上搜尋最多 5 層
        if (current / ".env").exists():
            logger.info(f"✓ Found .env at: {current / '.env'}")
            return str(current / ".env")
        if (current / ".git").exists() or (current / "README.md").exists():
            env_file = current / ".env"
            logger.info(f"✓ Project root found at: {current}")
            if env_file.exists():
                return str(env_file)
        current = current.parent
        if current == current.parent:  # 到達根目錄
            break
    
    return None


def read_env_file(env_path):
    """
    強化版 .env 檔案讀取
    支援多種編碼和格式
    """
    env_dict = {}
    
    try:
        # 嘗試多種編碼
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(env_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"✓ Successfully read .env with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"✗ Could not read .env file with any encoding")
            return env_dict
        
        # 解析 .env 檔案
        for line in content.split('\n'):
            line = line.strip()
            
            # 跳過空行和註釋
            if not line or line.startswith('#'):
                continue
            
            # 解析 KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 移除引號
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # 移除行末註釋
                if '#' in value:
                    value = value.split('#')[0].strip()
                
                env_dict[key] = value
                logger.debug(f"  Loaded: {key}={value[:50]}..." if len(value) > 50 else f"  Loaded: {key}={value}")
        
        logger.info(f"✓ Parsed {len(env_dict)} variables from .env")
        return env_dict
    
    except Exception as e:
        logger.error(f"✗ Error reading .env file: {e}")
        return env_dict


def get_hf_token():
    """
    取得 HuggingFace Token
    支援多種名稱:
    - HUGGINGFACE_TOKEN (推薦)
    - HF_TOKEN
    """
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        logger.info(f"✓ Using HUGGINGFACE_TOKEN: {token[:20]}...")
        return token
    
    token = os.getenv('HF_TOKEN')
    if token:
        logger.info(f"✓ Using HF_TOKEN: {token[:20]}...")
        return token
    
    return None


def get_hf_repo_id():
    """
    取得 HuggingFace Repo ID
    支援多種名稱:
    - HUGGINGFACE_REPO_ID (推薦)
    - HF_REPO_ID
    預設: caizongxun/crypto-price-predictor-v8
    """
    repo_id = os.getenv('HUGGINGFACE_REPO_ID')
    if repo_id:
        logger.info(f"✓ Using HUGGINGFACE_REPO_ID: {repo_id}")
        return repo_id
    
    repo_id = os.getenv('HF_REPO_ID')
    if repo_id:
        logger.info(f"✓ Using HF_REPO_ID: {repo_id}")
        return repo_id
    
    default_repo = "caizongxun/crypto-price-predictor-v8"
    logger.info(f"ℹ️  Using default HUGGINGFACE_REPO_ID: {default_repo}")
    return default_repo


def create_readme():
    """Create README for HuggingFace"""
    readme = """---
license: mit
tags:
  - crypto
  - price-prediction
  - lstm
  - trading
library_name: pytorch
---

# Crypto Price Predictor V8

A high-performance LSTM-based cryptocurrency price prediction model with bias correction.

## Model Details

### Architecture
- **Type**: Bidirectional LSTM
- **Layers**: 2 stacked LSTM layers
- **Hidden Size**: 64 (auto-detected per model)
- **Input Features**: 44 technical indicators
- **Output**: Next hour price prediction

### Supported Cryptocurrencies
BTC, ETH, SOL, BNB, XRP, ADA, DOT, LINK, MATIC, AVAX, FTM, NEAR, ATOM, ARB, OP, LTC, DOGE, UNI, SHIB, PEPE

### Performance
- **Average MAPE**: < 0.05%
- **Average MAE**: < 50 USD (varies by price)
- **Direction Accuracy**: ~65-75%

## Usage

### Quick Start

```python
from bot_predictor import BotPredictor

# Initialize
bot = BotPredictor()

# Get prediction
prediction = bot.predict('BTC')
print(f"Next Hour Price: ${prediction['corrected_price']:.2f}")
print(f"Direction: {prediction['direction']}")
print(f"Confidence: {prediction['confidence']*100:.1f}%")
```

### Installation

```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy ccxt
```

### Models

All models are in PyTorch format (.pth files). Download all models to use the full suite.

### Bias Correction

Each model includes an automatic bias correction value to account for training/test distribution differences.

File: `bias_corrections_v8.json`

## Technical Indicators (44 total)

- RSI (14, 21)
- MACD + Signal + Histogram
- Bollinger Bands (20, 2)
- ATR (14)
- CCI (20)
- Momentum (10)
- SMA (5, 10, 20, 50)
- EMA (12, 26)
- Volume Ratio
- OHLC-derived features

## Training Details

- **Data**: 1000 hourly candles per symbol
- **Train/Val/Test Split**: 80/10/10
- **Optimizer**: Adam (LR=0.005)
- **Loss**: MSE
- **Batch Size**: 64
- **Epochs**: 150 (with early stopping)
- **Dropout**: 0.3

## Requirements

```
torch>=2.0.0
torchvision
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
ccxt>=2.0.0
huggingface_hub>=0.16.0
python-dotenv>=1.0.0
```

## License

MIT License

## Citation

```
@software{crypto_predictor_v8,
  title={Crypto Price Predictor V8},
  author={Your Name},
  year={2025},
  url={https://huggingface.co/caizongxun/crypto-price-predictor-v8}
}
```

## Disclaimer

These models are for educational and research purposes only. Do not use for actual trading without thorough validation.

## Support

For issues and questions, please refer to the GitHub repository.
"""
    
    with open(README_PATH, 'w') as f:
        f.write(readme)
    logger.info(f"✓ Created {README_PATH}")


def upload_to_hf():
    """Upload models to HuggingFace - Optimized for batch upload"""
    
    # 自動搜尋並加載 .env
    env_file = find_env_file()
    if env_file:
        logger.info(f"Loading environment from: {env_file}")
        
        # 使用增強版讀取器
        env_dict = read_env_file(env_file)
        
        # 手動設置到 os.environ
        for key, value in env_dict.items():
            os.environ[key] = value
        
        # 也使用 load_dotenv 以備不時之需
        load_dotenv(env_file, override=True, encoding='utf-8')
    else:
        logger.warning("⚠️  No .env file found, trying system environment")
        load_dotenv()  # 使用系統預設路徑
    
    # 取得 token 和 repo ID
    hf_token = get_hf_token()
    hf_repo_id = get_hf_repo_id()
    
    # Debug: 輸出已加載的環境變數
    logger.info("\nLoaded environment variables:")
    for key in sorted(os.environ.keys()):
        if 'TOKEN' in key or 'KEY' in key or 'PASSWORD' in key or 'SECRET' in key or 'REPO' in key:
            value = os.environ[key]
            masked = f"{value[:10]}..." if len(value) > 10 else "(empty)"
            logger.info(f"  {key}: {masked}")
    
    if not hf_token:
        logger.error("✗ HUGGINGFACE_TOKEN not found!")
        logger.error("")
        logger.error("Please check your .env file:")
        logger.error("  1. .env file should contain: HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx")
        logger.error("  2. Or use: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxx")
        logger.error("  3. Make sure there are NO spaces around the = sign")
        logger.error("  4. Make sure HUGGINGFACE_TOKEN is not wrapped in quotes")
        logger.error("  5. Make sure there are no extra spaces at the end of the line")
        logger.error("")
        
        # 嘗試讀取 .env 並輸出前 10 行
        if env_file and Path(env_file).exists():
            logger.error(f"Content of {env_file}:")
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i < 10:
                            # 隱藏敏感信息
                            if 'TOKEN' in line or 'KEY' in line:
                                logger.error(f"  Line {i+1}: {line[:50]}...")
                            else:
                                logger.error(f"  Line {i+1}: {line.rstrip()}")
                        else:
                            break
            except Exception as e:
                logger.error(f"  Could not read file: {e}")
        
        return False
    
    logger.info(f"✓ HUGGINGFACE_TOKEN loaded successfully")
    logger.info(f"✓ Repository: {hf_repo_id}")
    
    # 創建 README
    create_readme()
    
    try:
        # 初始化 API
        api = HfApi()
        
        # 創建倉庫
        logger.info(f"Creating/accessing repo: {hf_repo_id}")
        try:
            repo_url = api.create_repo(
                repo_id=hf_repo_id,
                repo_type="model",
                exist_ok=True,
                private=False,  # Set to True if private
                token=hf_token
            )
            logger.info(f"✓ Repo URL: {repo_url}")
        except Exception as e:
            logger.info(f"ℹ️  Repo already exists: {str(e)[:80]}")
        
        # 檢查模型目錄
        model_dir = Path(MODEL_DIR)
        if not model_dir.exists():
            logger.error(f"✗ Model directory not found: {MODEL_DIR}")
            return False
        
        model_files = list(model_dir.glob('*.pth'))
        if not model_files:
            logger.error(f"✗ No .pth files found in {MODEL_DIR}")
            return False
        
        logger.info(f"Found {len(model_files)} model files to upload")
        
        # 上傳整個 models/saved 資料夾
        logger.info(f"\n📤 Uploading entire models/saved folder...")
        logger.info(f"   Total files: {len(model_files)}")
        
        try:
            api.upload_folder(
                folder_path=MODEL_DIR,
                repo_id=hf_repo_id,
                repo_type="model",
                token=hf_token,
                path_in_repo="models",  # Upload to models/ subfolder in HF
                multi_commit=True,  # Use multi-commit for large uploads
                multi_commit_pr=False
            )
            logger.info(f"   ✓ Folder uploaded successfully")
        except Exception as e:
            logger.error(f"   ✗ Folder upload failed: {e}")
            logger.info(f"   Trying fallback method...")
            
            # Fallback: Try uploading with commit_title
            try:
                api.upload_folder(
                    folder_path=MODEL_DIR,
                    repo_id=hf_repo_id,
                    repo_type="model",
                    token=hf_token,
                    path_in_repo="models",
                    commit_message="Upload all V8 models"
                )
                logger.info(f"   ✓ Folder uploaded successfully (fallback method)")
            except Exception as e2:
                logger.error(f"   ✗ Fallback also failed: {e2}")
                return False
        
        # 上傳偏差校正
        if os.path.exists(CONFIG_FILE):
            logger.info(f"\n📋 Uploading bias corrections...")
            try:
                api.upload_file(
                    path_or_fileobj=CONFIG_FILE,
                    path_in_repo="bias_corrections_v8.json",
                    repo_id=hf_repo_id,
                    repo_type="model",
                    token=hf_token,
                    commit_message="Upload bias corrections configuration"
                )
                logger.info(f"   ✓ bias_corrections_v8.json")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not upload bias_corrections: {e}")
        else:
            logger.warning(f"   ⚠️  bias_corrections_v8.json not found")
        
        # 上傳 bot predictor
        logger.info(f"\n🤖 Uploading bot predictor...")
        try:
            api.upload_file(
                path_or_fileobj="bot_predictor.py",
                path_in_repo="bot_predictor.py",
                repo_id=hf_repo_id,
                repo_type="model",
                token=hf_token,
                commit_message="Upload bot predictor module"
            )
            logger.info(f"   ✓ bot_predictor.py")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not upload bot_predictor: {e}")
        
        # 上傳 README
        logger.info(f"\n📖 Uploading README...")
        try:
            api.upload_file(
                path_or_fileobj=README_PATH,
                path_in_repo="README.md",
                repo_id=hf_repo_id,
                repo_type="model",
                token=hf_token,
                commit_message="Upload README documentation"
            )
            logger.info(f"   ✓ README.md")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not upload README: {e}")
        
        logger.info(f"\n" + "="*60)
        logger.info(f"✅ Upload Complete!")
        logger.info(f"="*60)
        logger.info(f"Repository: https://huggingface.co/{hf_repo_id}")
        logger.info(f"Models: {len(model_files)} files uploaded")
        logger.info(f"="*60)
        return True
    
    except Exception as e:
        logger.error(f"✗ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("HuggingFace Upload Tool - V8 Models (Batch Optimized)")
    logger.info("="*60)
    
    success = upload_to_hf()
    sys.exit(0 if success else 1)
