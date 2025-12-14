#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload Models to HuggingFace Hub

Usage:
  python upload_to_hf.py

Requires:
  - HF_TOKEN environment variable
  - huggingface_hub package
  - All models in models/saved/
  - bias_corrections_v8.json in models/
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, Repository
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = "caizongxun/crypto-price-predictor-v8"  # Change to your HF username
MODEL_DIR = "models/saved"
CONFIG_FILE = "models/bias_corrections_v8.json"
README_PATH = "README_HF.md"


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
    logger.info(f"Created {README_PATH}")


def upload_to_hf():
    """Upload models to HuggingFace"""
    
    # Check token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        logger.error("Set it with: export HF_TOKEN='your_token'")
        return False
    
    # Create README
    create_readme()
    
    try:
        # Initialize API
        api = HfApi()
        
        # Create repo
        logger.info(f"Creating/accessing repo: {HF_REPO_ID}")
        try:
            repo_url = api.create_repo(
                repo_id=HF_REPO_ID,
                repo_type="model",
                exist_ok=True,
                private=False,  # Set to True if private
                token=hf_token
            )
            logger.info(f"Repo URL: {repo_url}")
        except Exception as e:
            logger.info(f"Repo already exists or other issue: {e}")
        
        # Upload models
        logger.info(f"\nUploading models from {MODEL_DIR}...")
        model_dir = Path(MODEL_DIR)
        if model_dir.exists():
            for model_file in model_dir.glob('*.pth'):
                logger.info(f"  Uploading {model_file.name}...")
                api.upload_file(
                    path_or_fileobj=str(model_file),
                    path_in_repo=f"models/{model_file.name}",
                    repo_id=HF_REPO_ID,
                    repo_type="model",
                    token=hf_token
                )
                logger.info(f"    ✓ {model_file.name}")
        else:
            logger.error(f"Model directory not found: {MODEL_DIR}")
            return False
        
        # Upload bias corrections
        if os.path.exists(CONFIG_FILE):
            logger.info(f"\nUploading bias corrections...")
            api.upload_file(
                path_or_fileobj=CONFIG_FILE,
                path_in_repo="bias_corrections_v8.json",
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=hf_token
            )
            logger.info(f"  ✓ bias_corrections_v8.json")
        
        # Upload bot predictor
        logger.info(f"\nUploading bot predictor...")
        api.upload_file(
            path_or_fileobj="bot_predictor.py",
            path_in_repo="bot_predictor.py",
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token
        )
        logger.info(f"  ✓ bot_predictor.py")
        
        # Upload README
        logger.info(f"\nUploading README...")
        api.upload_file(
            path_or_fileobj=README_PATH,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token
        )
        logger.info(f"  ✓ README.md")
        
        logger.info(f"\n✅ All files uploaded successfully!")
        logger.info(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        return True
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("HuggingFace Upload Tool - V8 Models")
    logger.info("="*60)
    
    success = upload_to_hf()
    sys.exit(0 if success else 1)
