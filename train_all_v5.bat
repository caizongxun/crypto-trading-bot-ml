@echo off
REM ============================================================================
REM V5 訓練批量脚本 - 方向分類模式
REM ============================================================================
REM 特色：
  REM - 方向二分類（上漢/下跌）
  REM - CrossEntropyLoss + 簡配類別權重
  REM - 沒有趨児的配置
  REM - 恒定一道架構

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 V5 - 方向分類版本
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
set EPOCHS=150
set LR=0.0005
set BATCH_SIZE=16
set DEVICE=cuda

echo   Epochs: %EPOCHS%
echo   Learning Rate: %LR%
echo   Batch Size: %BATCH_SIZE%
echo   Device: %DEVICE%
echo.
echo 正在用 V5 (方向分類) 訓練市值前 20 的幣種...
echo   - 应推 比 V1/V2/V3 更成功
echo   - 只楷边的方向预测
echo.

REM 訓練每个幣種
set /a COUNT=0
set /a SUCCESS=0
set /a FAILED=0

for %%s in (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP) do (
    set /a COUNT=!COUNT!+1
    echo [!COUNT!/20] 訓練 %%s (v5 - direction classification)...
    python training/train_lstm_v5.py --symbol %%s --epochs %EPOCHS% --lr %LR% --batch-size %BATCH_SIZE% --device %DEVICE%
    
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
echo   - V5 檔索： models\saved\
echo   - V4 備份： models\backup_v4\
echo   - V5 結果： results\*_results_v5.json
echo.
echo 棧伊索統計：
echo   python visualize_results.py                    # 比較 v1-v5
echo   python visualize_results.py --symbol SOL       # 比較 SOL v1-v5
echo.
pause
