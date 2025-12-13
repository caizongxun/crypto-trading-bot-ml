@echo off
REM ============================================================================
REM 自動安裝並訓練 - 加密貨幣 LSTM 模型
REM ============================================================================
REM 用途：一鍵安裝 + 訓練
REM 平台：Windows
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 - 自動安裝並訓練
echo ================================================================================"
echo.

REM 檢查 Python
echo [1/3] 檢查 Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ✗ 找不到 Python！請先安裝 Python 3.9+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   ✓ Python %PYTHON_VERSION%

echo.
echo [2/3] 安裝依賴...

REM 刪除舊虛擬環境
if exist ".venv" (
    echo   刪除舊虛擬環境...
    rmdir /s /q .venv >nul 2>&1
)

REM 創建虛擬環境
echo   創建虛擬環境...
python -m venv .venv
call .venv\Scripts\activate.bat

REM 升級 pip
echo   升級 pip...
python -m pip install --upgrade pip setuptools wheel -q

REM 安裝 PyTorch GPU 版本
echo   安裝 PyTorch (GPU CUDA 12.4)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q

REM 安裝下际依賴
echo   安裝其他依賴...
pip install numpy pandas scikit-learn scipy -q
pip install ccxt -q
pip install pyyaml python-dotenv -q
pip install matplotlib seaborn pillow -q
pip install requests urllib3 discord.py -q

echo   ✓ 所有依賴已安裝

REM 驗證 GPU
echo.
echo [2.5/3] 驗證 GPU...
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"CPU Only\"}')" 2>nul
if errorlevel 1 (
    echo   ⚠ GPU 驗證失敗，後續用 CPU 訓練
)

REM 訓練
echo.
echo [3/3] 開始訓練市值前 20 的幣種...
echo ================================================================================"
echo.

python training/batch_train.py --no-git

echo.
echo ================================================================================"
echo ✓ 訓練完成！
echo ================================================================================"
echo.
echo 查看成果文件：
echo   - 日誌： logs\batch_train_*.log
echo   - 檔索： models\saved\
echo   - 結果： results\
echo.
pause
