@echo off
REM ============================================================================
REM 訓練市值前 20 的幣種 - 一个一个訓練 (不用 YAML)
REM ============================================================================
REM 用途：顺序訓練每个幣種
REM 平台：Windows
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 - 訓練市值前 20 的幣種
echo ================================================================================"
echo.

REM 检查虛擬環境
if not exist ".venv" (
    echo ❌ 找不到虛擬環境！請先運行 setup.bat 或 run.bat
    pause
    exit /b 1
)

echo 正在激活虛擬環境...
call .venv\Scripts\activate.bat
echo ✓ 虛擬環境已激活

echo.
echo 驗證 pyyaml...
python -c "import yaml; print('✓ pyyaml 接絣妨倩替打特')" 2>nul
if errorlevel 1 (
    echo   ⚠ pyyaml 不存在，正在安裝...
    pip install pyyaml -q
    echo   ✓ pyyaml 已安裝
)

echo.
echo 設置訓練參数：
REM 設置訓練參数
set EPOCHS=200
set DEVICE=cuda

echo   Epochs: %EPOCHS%
echo   Device: %DEVICE%
echo.
echo 正在訓練市值前 20 的幣種...
echo.

REM 訓練每个幣種
set /a COUNT=0
set /a SUCCESS=0
set /a FAILED=0

for %%s in (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP) do (
    set /a COUNT=!COUNT!+1
    echo [!COUNT!/20] 訓練 %%s...
    python training/train_lstm_v1.py --symbol %%s --epochs %EPOCHS% --device %DEVICE%
    
    if errorlevel 1 (
        set /a FAILED=!FAILED!+1
        echo   ✗ %%s 訓練失敗
    ) else (
        set /a SUCCESS=!SUCCESS!+1
        echo   ✓ %%s 訓練成功
    )
    echo.
)

echo ================================================================================"
echo 訓練完成！總結回軸
echo ================================================================================"
echo   成功: %SUCCESS%/20
echo   失敗: %FAILED%/20
echo.
echo 檔索位置：
echo   - models\saved\
echo   - results\
echo.
pause
