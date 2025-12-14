@echo off
REM ================================================================
REM V9 全幣種批量訓練脚本 (精度提升版)
REM ================================================================
REM 特訪: 更大网络 (256x3) + 更多技术指標 (60+) + 改進正則化
REM 訓練時間: 大約 2-3 小時 (每個約 8-10 分鐘)
REM ================================================================

echo ========================================
echo   V9 BATCH TRAINING - Enhanced Precision
echo   All 20 Symbols (256x3 Network)
echo ========================================
echo.
echo 開始時間: %date% %time%
echo.

REM 拉取最新程式碼
python -c "import subprocess; subprocess.run(['git', 'pull', 'origin', 'main'])"

echo.
echo ========== 開始訓練 V9 ==========
echo.

REM BTC
echo [1/20] Training BTC (256x3, 60+ indicators)...
python training/train_lstm_v9_precision.py --symbol BTC --epochs 200 --batch-size 16 --lr 0.0005

REM ETH
echo [2/20] Training ETH...
python training/train_lstm_v9_precision.py --symbol ETH --epochs 200 --batch-size 16 --lr 0.0005

REM BNB
echo [3/20] Training BNB...
python training/train_lstm_v9_precision.py --symbol BNB --epochs 200 --batch-size 16 --lr 0.0005

REM XRP
echo [4/20] Training XRP...
python training/train_lstm_v9_precision.py --symbol XRP --epochs 200 --batch-size 16 --lr 0.0005

REM ADA
echo [5/20] Training ADA...
python training/train_lstm_v9_precision.py --symbol ADA --epochs 200 --batch-size 16 --lr 0.0005

REM DOGE
echo [6/20] Training DOGE...
python training/train_lstm_v9_precision.py --symbol DOGE --epochs 200 --batch-size 16 --lr 0.0005

REM SOL
echo [7/20] Training SOL...
python training/train_lstm_v9_precision.py --symbol SOL --epochs 200 --batch-size 16 --lr 0.0005

REM POLKADOT
echo [8/20] Training DOT...
python training/train_lstm_v9_precision.py --symbol DOT --epochs 200 --batch-size 16 --lr 0.0005

REM AVAX
echo [9/20] Training AVAX...
python training/train_lstm_v9_precision.py --symbol AVAX --epochs 200 --batch-size 16 --lr 0.0005

REM LINK
echo [10/20] Training LINK...
python training/train_lstm_v9_precision.py --symbol LINK --epochs 200 --batch-size 16 --lr 0.0005

REM UNI
echo [11/20] Training UNI...
python training/train_lstm_v9_precision.py --symbol UNI --epochs 200 --batch-size 16 --lr 0.0005

REM LTC
echo [12/20] Training LTC...
python training/train_lstm_v9_precision.py --symbol LTC --epochs 200 --batch-size 16 --lr 0.0005

REM MATIC
echo [13/20] Training MATIC...
python training/train_lstm_v9_precision.py --symbol MATIC --epochs 200 --batch-size 16 --lr 0.0005

REM ARB
echo [14/20] Training ARB...
python training/train_lstm_v9_precision.py --symbol ARB --epochs 200 --batch-size 16 --lr 0.0005

REM OPTIMISM
echo [15/20] Training OP...
python training/train_lstm_v9_precision.py --symbol OP --epochs 200 --batch-size 16 --lr 0.0005

REM ATOM
echo [16/20] Training ATOM...
python training/train_lstm_v9_precision.py --symbol ATOM --epochs 200 --batch-size 16 --lr 0.0005

REM FTM
echo [17/20] Training FTM...
python training/train_lstm_v9_precision.py --symbol FTM --epochs 200 --batch-size 16 --lr 0.0005

REM NEAR
echo [18/20] Training NEAR...
python training/train_lstm_v9_precision.py --symbol NEAR --epochs 200 --batch-size 16 --lr 0.0005

REM PEPE
echo [19/20] Training PEPE...
python training/train_lstm_v9_precision.py --symbol PEPE --epochs 200 --batch-size 16 --lr 0.0005

REM SHIB
echo [20/20] Training SHIB...
python training/train_lstm_v9_precision.py --symbol SHIB --epochs 200 --batch-size 16 --lr 0.0005

echo.
echo ========================================
echo   訓練完成!
echo 結束時間: %date% %time%
echo ========================================
echo.
echo 查看 V9 結果:
python visualize_predictions_v8.py

pause
