#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Models from HuggingFace Hub for VM Deployment

Usage:
  python download_from_hf.py

Sets up:
  - models/saved/ with all .pth model files
  - bias_corrections_v8.json
  - bot_predictor.py

Requires:
  - huggingface_hub package
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = "caizongxun/crypto-price-predictor-v8"  # Change if needed
MODEL_DIR = "models/saved"


def ensure_directories():
    """Ensure required directories exist"""
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory: {MODEL_DIR}")


def download_models_from_hf():
    """Download all model files from HuggingFace"""
    
    try:
        logger.info(f"\nFetching file list from {HF_REPO_ID}...")
        
        # Get all files in repo
        files = list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        
        # Filter model files
        model_files = [f for f in files if f.startswith('models/') and f.endswith('.pth')]
        
        if not model_files:
            logger.warning("No model files found in repository")
            return False
        
        logger.info(f"Found {len(model_files)} model files")
        
        # Download each model
        logger.info(f"\nDownloading models...")
        for model_file in model_files:
            file_name = os.path.basename(model_file)
            save_path = os.path.join(MODEL_DIR, file_name)
            
            logger.info(f"  Downloading {file_name}...")
            
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=model_file,
                    repo_type="model",
                    local_dir=".",
                    force_download=False
                )
                logger.info(f"    ✓ {file_name}")
            except Exception as e:
                logger.error(f"    ❌ Failed to download {file_name}: {e}")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error listing repository: {e}")
        return False


def download_config_files():
    """Download configuration and supporting files"""
    
    config_files = [
        ('bias_corrections_v8.json', 'bias_corrections_v8.json'),
        ('bot_predictor.py', 'bot_predictor.py'),
    ]
    
    logger.info(f"\nDownloading configuration files...")
    
    for remote_file, local_file in config_files:
        try:
            logger.info(f"  Downloading {remote_file}...")
            
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=remote_file,
                repo_type="model",
                local_dir=".",
                force_download=False
            )
            
            logger.info(f"    ✓ {local_file}")
        
        except Exception as e:
            logger.warning(f"    ❌ Could not download {remote_file}: {e}")
            logger.warning(f"       You may need to download it manually")
    
    return True


def verify_downloads():
    """Verify all downloads are complete"""
    
    logger.info(f"\nVerifying downloads...")
    
    # Check model directory
    model_dir_path = Path(MODEL_DIR)
    if not model_dir_path.exists():
        logger.error(f"Model directory not found: {MODEL_DIR}")
        return False
    
    model_files = list(model_dir_path.glob('*.pth'))
    if not model_files:
        logger.error(f"No model files found in {MODEL_DIR}")
        return False
    
    logger.info(f"  ✓ Found {len(model_files)} model files")
    
    # Check config file
    if os.path.exists('bias_corrections_v8.json'):
        logger.info(f"  ✓ bias_corrections_v8.json found")
    else:
        logger.warning(f"  ❌ bias_corrections_v8.json not found (will use defaults)")
    
    # Check bot predictor
    if os.path.exists('bot_predictor.py'):
        logger.info(f"  ✓ bot_predictor.py found")
    else:
        logger.warning(f"  ❌ bot_predictor.py not found (required for Discord bot)")
    
    return len(model_files) > 0


def main():
    """Main download procedure"""
    
    logger.info("="*60)
    logger.info("HuggingFace Model Downloader - V8 Models")
    logger.info("="*60)
    logger.info(f"Repository: {HF_REPO_ID}")
    logger.info(f"Target Directory: {MODEL_DIR}")
    
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
        logger.info("  2. Run: python bot_predictor.py to test predictions")
        logger.info("  3. Train new models: python train_v8_models.py")
        return True
    else:
        logger.error("\nDownload verification failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
