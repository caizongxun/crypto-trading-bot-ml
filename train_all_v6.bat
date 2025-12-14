@echo off
REM ============================================================================
REM V6 訓練批量脚本 - 方向專家模式
REM ============================================================================
REM 特訓：
  REM - 3 級方向分類（下跌/中性/上漲）
  REM - Focal Loss 解決類別不衡
  REM - 加權採樣器 准點不衡
  REM - 最後建識單元架構
REM 目標：方向準確度 70% +

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 V6 - 方向專家模式
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
REM 訓練參数
set EPOCHS=200
set LR=0.001
set BATCH_SIZE=32
set THRESHOLD=0.05
set DEVICE=cuda

echo   Epochs: %EPOCHS%
echo   Learning Rate: %LR%
echo   Batch Size: %BATCH_SIZE%
echo   Direction Threshold: %THRESHOLD% (5%%)
echo   Device: %DEVICE%
echo.
echo 正在用 V6 (方向專家) 訓練市值前 20 的幣種...
echo   - 此次稬紀的穼非一怎模式：
echo   - 3級方向分類（下跌/中性/上漲）
echo   - Focal Loss + 加權採樣
echo   - 目標：70%+ 準確度
echo.

REM 訓練每个幣種
set /a COUNT=0
set /a SUCCESS=0
set /a FAILED=0

for %%s in (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP) do (
    set /a COUNT=!COUNT!+1
    echo [!COUNT!/20] 訓練 %%s (v6 - direction expert)...
    python training/train_lstm_v6_direction_expert.py --symbol %%s --epochs %EPOCHS% --lr %LR% --batch-size %BATCH_SIZE% --threshold %THRESHOLD% --device %DEVICE%
    
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
echo   - V6 檔索： models\saved\
echo   - V5 備份： models\backup_v5\
echo   - V6 結果： results\*_results_v6.json
echo.
echo 棧伊索統計：
echo   python visualize_results.py                    # 比較 v1-v6
echo   python visualize_results.py --symbol SOL       # 比較 SOL v1-v6
echo.
pause
