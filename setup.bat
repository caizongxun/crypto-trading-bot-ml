@echo off
REM ============================================================================
REM 完整設置腳本 - 加密貨幣價格預測模型 (Windows)
REM ============================================================================
REM 用途：一鍵設置虛擬環境並安裝所有依賴
REM 平台：Windows (cmd.exe or PowerShell)
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 - 完整設置腳本 (Windows)
echo ================================================================================"
echo.

REM 檢查 Python 版本
echo [1/5] 檢查 Python 版本...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ✗ 找不到 Python！請先安裝 Python 3.9+
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   ✓ Python %PYTHON_VERSION%

REM 刪除舊虛擬環境
echo.
echo [2/5] 準備虛擬環境...
if exist ".venv" (
    echo   ⚠ 找到舊虛擬環境，正在刪除...
    rmdir /s /q .venv >nul 2>&1
    echo   ✓ 已刪除舊虛擬環境
) else (
    echo   ✓ 虛擬環境不存在，將新建
)

REM 創建新虛擬環境
echo   正在創建新虛擬環境...
python -m venv .venv
echo   ✓ 虛擬環境已創建

REM 激活虛擬環境
echo.
echo [3/5] 激活虛擬環境...
call .venv\Scripts\activate.bat
echo   ✓ 虛擬環境已激活

REM 升級 pip
echo.
echo [4/5] 升級 pip 和基礎工具...
python -m pip install --upgrade pip setuptools wheel -q
echo   ✓ pip 已升級

REM 安裝依賴
echo.
echo [5/5] 安裝依賴套件...
echo   安裝 PyTorch (GPU 支持)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
echo   ✓ PyTorch 已安裝

echo   安裝數據處理和科學計算...
pip install numpy pandas scikit-learn scipy -q
echo   ✓ 數據處理庫已安裝

echo   安裝交易所 API...
pip install ccxt -q
echo   ✓ CCXT 已安裝

echo   安裝配置和工具...
pip install pyyaml python-dotenv -q
echo   ✓ 配置工具已安裝

echo   安裝可視化工具...
pip install matplotlib seaborn pillow -q
echo   ✓ 可視化工具已安裝

echo   安裝網絡工具...
pip install requests urllib3 -q
echo   ✓ 網絡工具已安裝

echo   安裝 Discord Bot (可選)...
pip install discord.py -q
echo   ✓ Discord Bot 已安裝

REM 驗證關鍵套件
echo.
echo [驗證] 檢查關鍵套件...
echo   檢查 PyTorch...
python -c "import torch; print(f'  ✓ {torch.__version__}')"

echo   檢查 YAML...
python -c "import yaml; print(f'  ✓ {yaml.__version__}')"

echo   檢查 CCXT...
python -c "import ccxt; print(f'  ✓ {ccxt.__version__}')"

echo   檢查 pandas...
python -c "import pandas; print(f'  ✓ {pandas.__version__}')"

echo   檢查 scikit-learn...
python -c "import sklearn; print(f'  ✓ {sklearn.__version__}')"

echo.
echo ================================================================================"
echo ✓ 設置完成！
echo ================================================================================"
echo.
echo 下一步：
echo   1. 甦訓練單個模型：
echo      python training\train_lstm_v1.py --symbol SOL --epochs 200
echo.
echo   2. 批量訓練市值前 20 的幣種：
echo      python training\batch_train.py --no-git
echo.
echo   3. 或訓練指定幣種：
echo      python training\batch_train.py --symbols SOL,BTC,ETH
echo.
pause
