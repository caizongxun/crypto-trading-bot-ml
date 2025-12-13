#!/bin/bash

# ============================================================================
# 完整設置腳本 - 加密貨幣價格預測模型
# ============================================================================
# 用途：一鍵設置虛擬環境並安裝所有依賴
# 平台：Linux / macOS
# ============================================================================

set -e  # 任何錯誤就停止

echo ""
echo "================================================================================"
echo "加密貨幣 LSTM 模型 - 完整設置腳本"
echo "================================================================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# 檢查 Python 版本
echo "${YELLOW}[1/5] 檢查 Python 版本...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "${RED}  ✗ 找不到 Python 3！請先安裝 Python 3.9+${NC}"
    exit 1
fi

# 刪除舊虛擬環境（如果存在）
echo ""
echo "${YELLOW}[2/5] 準備虛擬環境...${NC}"
if [ -d ".venv" ]; then
    echo "  ⚠ 找到舊虛擬環境，正在刪除..."
    rm -rf .venv
    echo "  ✓ 已刪除舊虛擬環境"
else
    echo "  ✓ 虛擬環境不存在，將新建"
fi

# 創建新虛擬環境
echo "  正在創建新虛擬環境..."
python3 -m venv .venv
echo "  ✓ 虛擬環境已創建"

# 激活虛擬環境
echo ""
echo "${YELLOW}[3/5] 激活虛擬環境...${NC}"
source .venv/bin/activate
echo "  ✓ 虛擬環境已激活"

# 升級 pip
echo ""
echo "${YELLOW}[4/5] 升級 pip 和基礎工具...${NC}"
pip install --upgrade pip setuptools wheel
echo "  ✓ pip 已升級"

# 安裝依賴
echo ""
echo "${YELLOW}[5/5] 安裝依賴套件...${NC}"
echo "  安裝 PyTorch (GPU 支持)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
echo "  ✓ PyTorch 已安裝"

echo "  安裝數據處理和科學計算..."
pip install numpy pandas scikit-learn scipy -q
echo "  ✓ 數據處理庫已安裝"

echo "  安裝交易所 API..."
pip install ccxt -q
echo "  ✓ CCXT 已安裝"

echo "  安裝配置和工具..."
pip install pyyaml python-dotenv -q
echo "  ✓ 配置工具已安裝"

echo "  安裝可視化工具..."
pip install matplotlib seaborn pillow -q
echo "  ✓ 可視化工具已安裝"

echo "  安裝網絡工具..."
pip install requests urllib3 -q
echo "  ✓ 網絡工具已安裝"

echo "  安裝 Discord Bot (可選)..."
pip install discord.py -q
echo "  ✓ Discord Bot 已安裝"

# 驗證關鍵套件
echo ""
echo "${YELLOW}[驗證] 檢查關鍵套件...${NC}"

echo -n "  檢查 PyTorch... "
python -c "import torch; print(f'✓ {torch.__version__}')"

echo -n "  檢查 YAML... "
python -c "import yaml; print(f'✓ {yaml.__version__}')"

echo -n "  檢查 CCXT... "
python -c "import ccxt; print(f'✓ {ccxt.__version__}')"

echo -n "  檢查 pandas... "
python -c "import pandas; print(f'✓ {pandas.__version__}')"

echo -n "  檢查 scikit-learn... "
python -c "import sklearn; print(f'✓ {sklearn.__version__}')"

echo ""
echo "${GREEN}================================================================================"
echo "✓ 設置完成！"
echo "${GREEN}================================================================================${NC}"
echo ""
echo "${GREEN}下一步：${NC}"
echo "  1. 激活虛擬環境："
echo "     source .venv/bin/activate"
echo ""
echo "  2. 訓練單個模型："
echo "     python training/train_lstm_v1.py --symbol SOL --epochs 200"
echo ""
echo "  3. 批量訓練市值前 20 的幣種："
echo "     python training/batch_train.py --no-git"
echo ""
echo "  4. 或訓練指定幣種："
echo "     python training/batch_train.py --symbols SOL,BTC,ETH"
echo ""
