@echo off
REM ============================================================================
REM V3 訓練市值前 20 的幣種 - 平衡優化版本
REM ============================================================================
REM 优化：70% 價格損失 + 30% 方向損失
REM 平台：Windows
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 V3 - 平衡優化
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
echo 設置訓練參数：
REM 設置訓練參数
set EPOCHS=200
set LR=0.0005
set DEVICE=cuda
set LOAD_V1=0

echo   Epochs: %EPOCHS% (恢復 v1 的數量)
echo   Learning Rate: %LR% (恢復 v1 的学习率)
echo   Device: %DEVICE%
echo   Load v1 models: No (从新訓笺)
echo   Loss: 70%% Price + 30%% Direction (平衡优化)
echo.
echo 正在用 v3 訓練市值前 20 的幣種...
echo.

REM 訓練每个幣種
set /a COUNT=0
set /a SUCCESS=0
set /a FAILED=0

for %%s in (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP) do (
    set /a COUNT=!COUNT!+1
    echo [!COUNT!/20] 訓練 %%s (v3 - 平衡優化)...
    python training/train_lstm_v3.py --symbol %%s --epochs %EPOCHS% --lr %LR% --device %DEVICE%
    
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
echo   - V3 檔索： models\saved\
echo   - V2 備份： models\backup_v2\
echo   - V1 備份： models\backup_v1\
echo   - V3 結果： results\*_results_v3.json
echo.
echo 棧伊一、二：
echo   python visualize_results.py            # 比較所有三個版本
echo   python visualize_results.py --symbol SOL  # 比較 SOL
echo.
pause
