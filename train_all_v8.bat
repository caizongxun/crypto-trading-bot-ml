@echo off
REM ================================================================
REM V8 全幣種批量訓練脚本
REM ================================================================
REM 特訪: 純價格回歸 (V8 穩定版本)
REM 訓練時間: 大約 1.5-2 小時 (每個約 5-6 分鐘)
REM ================================================================

echo ========================================
echo   V8 BATCH TRAINING - All 20 Symbols
echo ========================================
echo.
echo 開始時間: %date% %time%
echo.

REM 拉取最新程式碼
python -c "import subprocess; subprocess.run(['git', 'pull', 'origin', 'main'])"

echo.
echo ========== 開始訓練 ==========
echo.

REM BTC
echo [1/20] Training BTC...
python training/train_lstm_v8_pure_regression.py --symbol BTC --epochs 150 --batch-size 16

REM ETH
echo [2/20] Training ETH...
python training/train_lstm_v8_pure_regression.py --symbol ETH --epochs 150 --batch-size 16

REM BNB
echo [3/20] Training BNB...
python training/train_lstm_v8_pure_regression.py --symbol BNB --epochs 150 --batch-size 16

REM XRP
echo [4/20] Training XRP...
python training/train_lstm_v8_pure_regression.py --symbol XRP --epochs 150 --batch-size 16

REM ADA
echo [5/20] Training ADA...
python training/train_lstm_v8_pure_regression.py --symbol ADA --epochs 150 --batch-size 16

REM DOGE
echo [6/20] Training DOGE...
python training/train_lstm_v8_pure_regression.py --symbol DOGE --epochs 150 --batch-size 16

REM SOL
echo [7/20] Training SOL...
python training/train_lstm_v8_pure_regression.py --symbol SOL --epochs 150 --batch-size 16

REM POLKADOT
echo [8/20] Training DOT...
python training/train_lstm_v8_pure_regression.py --symbol DOT --epochs 150 --batch-size 16

REM AVAX
echo [9/20] Training AVAX...
python training/train_lstm_v8_pure_regression.py --symbol AVAX --epochs 150 --batch-size 16

REM LINK
echo [10/20] Training LINK...
python training/train_lstm_v8_pure_regression.py --symbol LINK --epochs 150 --batch-size 16

REM UNI
echo [11/20] Training UNI...
python training/train_lstm_v8_pure_regression.py --symbol UNI --epochs 150 --batch-size 16

REM LTC
echo [12/20] Training LTC...
python training/train_lstm_v8_pure_regression.py --symbol LTC --epochs 150 --batch-size 16

REM MATIC
echo [13/20] Training MATIC...
python training/train_lstm_v8_pure_regression.py --symbol MATIC --epochs 150 --batch-size 16

REM ARB
echo [14/20] Training ARB...
python training/train_lstm_v8_pure_regression.py --symbol ARB --epochs 150 --batch-size 16

REM OPTIMISM
echo [15/20] Training OP...
python training/train_lstm_v8_pure_regression.py --symbol OP --epochs 150 --batch-size 16

REM ATOM
echo [16/20] Training ATOM...
python training/train_lstm_v8_pure_regression.py --symbol ATOM --epochs 150 --batch-size 16

REM FTM
echo [17/20] Training FTM...
python training/train_lstm_v8_pure_regression.py --symbol FTM --epochs 150 --batch-size 16

REM NEAR
echo [18/20] Training NEAR...
python training/train_lstm_v8_pure_regression.py --symbol NEAR --epochs 150 --batch-size 16

REM PEPE
echo [19/20] Training PEPE...
python training/train_lstm_v8_pure_regression.py --symbol PEPE --epochs 150 --batch-size 16

REM SHIB
echo [20/20] Training SHIB...
python training/train_lstm_v8_pure_regression.py --symbol SHIB --epochs 150 --batch-size 16

echo.
echo ========================================
echo   訓練完成!
necho 結束時間: %date% %time%
echo ========================================
echo.
echo 查看結果:
python visualize_results.py

pause
