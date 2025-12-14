#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Models from HuggingFace Hub for VM Deployment

Usage:
  python download_from_hf.py

Features:
  - Auto-finds .env file in project root and parent directories
  - Reads HF_TOKEN from .env file (optional, for private repos)
  - Downloads entire models/ folder
  - Downloads bias_corrections_v8.json
  - Downloads bot_predictor.py

Sets up:
  - models/saved/ with all .pth model files
  - bias_corrections_v8.json
  - bot_predictor.py

Requires:
  - .env file with HF_TOKEN (optional, unless repo is private)
  - huggingface_hub package
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = "caizongxun/crypto-price-predictor-v8"  # Change if needed
MODEL_DIR = "models/saved"


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


def ensure_directories():
    """Ensure required directories exist"""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Ensured directory: {MODEL_DIR}")


def download_models_from_hf():
    """Download all model files from HuggingFace"""
    
    hf_token = os.getenv('HF_TOKEN', None)  # Optional for public repos
    
    try:
        logger.info(f"\n📦 Downloading models from {HF_REPO_ID}...")
        
        # Get all files in repo
        files = list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token
        )
        
        # Filter model files
        model_files = [f for f in files if f.startswith('models/') and f.endswith('.pth')]
        
        if not model_files:
            logger.warning("⚠️  No model files found in repository")
            return False
        
        logger.info(f"Found {len(model_files)} model files")
        
        # Download entire models folder using snapshot_download (much faster)
        logger.info(f"\n📥 Downloading entire models/ folder...")
        
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                allow_patterns=["models/*.pth"],  # Only .pth files
                local_dir=".",
                token=hf_token
            )
            logger.info(f"   ✓ Models downloaded successfully")
        except Exception as e:
            logger.warning(f"   ⚠️  Snapshot download failed: {e}")
            logger.info(f"   Falling back to individual downloads...")
            
            # Fallback: Download individual files
            for model_file in model_files:
                file_name = os.path.basename(model_file)
                logger.info(f"   Downloading {file_name}...")
                
                try:
                    hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename=model_file,
                        repo_type="model",
                        local_dir=".",
                        token=hf_token
                    )
                    logger.info(f"      ✓ {file_name}")
                except Exception as e:
                    logger.error(f"      ✗ Failed: {e}")
                    return False
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Error listing repository: {e}")
        return False


def download_config_files():
    """Download configuration and supporting files"""
    
    config_files = [
        ('bias_corrections_v8.json', 'bias_corrections_v8.json'),
        ('bot_predictor.py', 'bot_predictor.py'),
    ]
    
    logger.info(f"\n📄 Downloading configuration files...")
    
    hf_token = os.getenv('HF_TOKEN', None)
    
    for remote_file, local_file in config_files:
        try:
            logger.info(f"  Downloading {remote_file}...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=remote_file,
                repo_type="model",
                local_dir=".",
                token=hf_token
            )
            
            logger.info(f"    ✓ {local_file}")
        
        except Exception as e:
            logger.warning(f"    ⚠️  Could not download {remote_file}: {e}")
            logger.warning(f"       You may need to download it manually")
    
    return True


def verify_downloads():
    """Verify all downloads are complete"""
    
    logger.info(f"\n🔍 Verifying downloads...")
    
    # Check model directory
    model_dir_path = Path(MODEL_DIR)
    if not model_dir_path.exists():
        logger.error(f"✗ Model directory not found: {MODEL_DIR}")
        return False
    
    model_files = list(model_dir_path.glob('*.pth'))
    if not model_files:
        logger.error(f"✗ No model files found in {MODEL_DIR}")
        return False
    
    logger.info(f"  ✓ Found {len(model_files)} model files")
    for mf in model_files[:3]:
        logger.info(f"     - {mf.name}")
    if len(model_files) > 3:
        logger.info(f"     ... and {len(model_files) - 3} more")
    
    # Check config file
    if os.path.exists('bias_corrections_v8.json'):
        logger.info(f"  ✓ bias_corrections_v8.json found")
    else:
        logger.warning(f"  ⚠️  bias_corrections_v8.json not found (will use defaults)")
    
    # Check bot predictor
    if os.path.exists('bot_predictor.py'):
        logger.info(f"  ✓ bot_predictor.py found")
    else:
        logger.warning(f"  ⚠️  bot_predictor.py not found (required for Discord bot)")
    
    return len(model_files) > 0


def main():
    """Main download procedure"""
    
    logger.info("="*60)
    logger.info("HuggingFace Model Downloader - V8 Models")
    logger.info("="*60)
    logger.info(f"Repository: {HF_REPO_ID}")
    logger.info(f"Target Directory: {MODEL_DIR}")
    
    # 自動搜尋並加載 .env
    env_file = find_env_file()
    if env_file:
        logger.info(f"Loading environment from: {env_file}")
        load_dotenv(env_file)
    else:
        logger.warning("⚠️  No .env file found, trying system environment")
        load_dotenv()  # 使用系統預設路徑
    
    hf_token = os.getenv('HF_TOKEN', None)
    if hf_token:
        logger.info(f"✓ HF_TOKEN loaded: {hf_token[:20]}...")
    else:
        logger.info(f"ℹ️  No HF_TOKEN found (OK for public repos)")
    
    # Step 1: Ensure directories
    ensure_directories()
    
    # Step 2: Download models
    if not download_models_from_hf():
        logger.error("Failed to download models")
        return False
    
    # Step 3: Download config files
    download_config_files()
    
    # Step 4: Verify
    if verify_downloads():
        logger.info("\n" + "="*60)
        logger.info("✅ Download Complete!")
        logger.info("="*60)
        logger.info("\nYou can now:")
        logger.info("  1. Use bot_predictor.py with your Discord bot")
        logger.info("  2. Run: python -c 'from bot_predictor import BotPredictor; bot = BotPredictor(); print(bot.predict(\"BTC\"))' to test")
        logger.info("  3. Train new models: python train_v8_models.py")
        logger.info("="*60)
        return True
    else:
        logger.error("\nDownload verification failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
