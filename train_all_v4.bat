@echo off
REM ============================================================================
REM V4 訓練市值前 20 的幣種 - LoRA 子訓練版本
REM ============================================================================
REM 特色：
  REM - 從 v1 檔索依上
  REM - LoRA 低秘秘雨子訓練
  REM - 三個輸出肠：價格 + 方向標簽 + 幅度
REM 平台：Windows
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ================================================================================"
echo 加密貨幣 LSTM 模型 V4 - LoRA 子訓練版本
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
set EPOCHS=150
set LR=0.0005
set LORA_RANK=4
set DEVICE=cuda

echo   Epochs: %EPOCHS%
echo   Learning Rate: %LR%
echo   LoRA Rank: %LORA_RANK%
echo   Device: %DEVICE%
echo.
echo 正在用 v4 (LoRA) 訓練市值前 20 的幣種...
echo   - 方向標簽輸出作為主警警
  echo   - LoRA 低細糛s子訓練
echo.

REM 訓練每个幣種
set /a COUNT=0
set /a SUCCESS=0
set /a FAILED=0

for %%s in (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP) do (
    set /a COUNT=!COUNT!+1
    echo [!COUNT!/20] 訓練 %%s (v4 - LoRA)...
    python training/train_lstm_v4.py --symbol %%s --epochs %EPOCHS% --lr %LR% --lora-rank %LORA_RANK% --device %DEVICE%
    
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
echo   - V4 檔索： models\saved\
echo   - V3 備份： models\backup_v3\
echo   - V4 結果： results\*_results_v4.json
echo.
echo 棧伊索統計：
echo   python visualize_results.py                    # 比較 v1-v4
echo   python visualize_results.py --symbol SOL       # 比較 SOL v1-v4
echo.
pause
