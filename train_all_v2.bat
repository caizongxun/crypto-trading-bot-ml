@echo off
REM ============================================================================
REM V2 訓練市值前 20 的幣種 - 方向準確度優化
REM ============================================================================
REM 基於 v1 檔索進行微調
REM 平台：Windows
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 V2 - 方向準確度優化
echo ================================================================================"
echo.

REM 检查虛擬環境
if not exist ".venv" (
    echo ❌ 找不到虛擬環境！請先運行 setup.bat
    pause
    exit /b 1
)

echo 正在激活虛擬環境...
call .venv\Scripts\activate.bat
echo ✓ 虛擬環境已激活

echo.
echo 設置訓練參數：
REM 設置訓練參数
set EPOCHS=100
set LR=0.0001
set DEVICE=cuda
set LOAD_V1=1

echo   Epochs: %EPOCHS% (低于 v1 克尋我)
echo   Learning Rate: %LR% (低于 v1 克尋我)
echo   Device: %DEVICE%
echo   Load v1 models: Yes
echo.
echo 正在用 v2 訓練市值前 20 的幣種...
echo (会自动打惫为 v1 的檔索至 models/backup_v1/)
echo.

REM 訓練每个幣種
set /a COUNT=0
set /a SUCCESS=0
set /a FAILED=0

for %%s in (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP) do (
    set /a COUNT=!COUNT!+1
    echo [!COUNT!/20] 訓練 %%s (v2 - 方向準確度優化)...
    python training/train_lstm_v2.py --symbol %%s --epochs %EPOCHS% --lr %LR% --device %DEVICE% --load-v1
    
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
echo   - V2 檔索： models\saved\
echo   - V1 備份eff1a️： models\backup_v1\
echo   - V2 結果： results\*_results_v2.json
echo   - V1 結果： results\*_results.json
echo.
echo 回調到 v1：
echo   python training/train_lstm_v1.py --symbol BTC
echo.
pause
