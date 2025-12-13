#!/bin/bash

# ============================================================================
# 自動安裝並訓練 - 加密貨幣 LSTM 模型
# ============================================================================
# 用途：一鍵安裝 + 訓練
# 平台：Linux / macOS
# ============================================================================

set -e

echo ""
echo "================================================================================"
echo "加密貨幣 LSTM 模型 - 自動安裝並訓練"
echo "================================================================================"
echo ""

# 檢查 Python
echo "[1/3] 檢查 Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "  ✗ 找不到 Python 3！請先安裝 Python 3.9+"
    exit 1
fi

echo ""
echo "[2/3] 安裝依賴..."

# 刪除舊虛擬環境
if [ -d ".venv" ]; then
    echo "  刪除舊虛擬環境..."
    rm -rf .venv
fi

# 創建虛擬環境
echo "  創建虛擬環境..."
python3 -m venv .venv
source .venv/bin/activate

# 升級 pip
echo "  升級 pip..."
pip install --upgrade pip setuptools wheel -q

# 安裝 PyTorch GPU 版本
echo "  安裝 PyTorch (GPU CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

# 安裝下际依賴
echo "  安裝其他依賴..."
pip install numpy pandas scikit-learn scipy -q
pip install ccxt -q
pip install pyyaml python-dotenv -q
pip install matplotlib seaborn pillow -q
pip install requests urllib3 discord.py -q

echo "  ✓ 所有依賴已安裝"

# 驗證 GPU
echo ""
echo "[2.5/3] 驗證 GPU..."
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU Only\"}')" 2>/dev/null || echo "  ⚠ GPU 驗證失敗，後續用 CPU 訓練"

# 訓練
echo ""
echo "[3/3] 開始訓練市值前 20 的幣種..."
echo "================================================================================"
echo ""

python training/batch_train.py --no-git

echo ""
echo "================================================================================"
echo "✓ 訓練完成！"
echo "================================================================================"
echo ""
echo "查看成果文件："
echo "  - 日誌： logs/batch_train_*.log"
echo "  - 檔索： models/saved/"
echo "  - 結果： results/"
echo ""
