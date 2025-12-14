#!/bin/bash

# ================================================================
# V8 全幣種批量訓練腳本
# ================================================================
# 特性: 純價格回歸 (V8 穩定版本)
# 訓練時間: 大約 1.5-2 小時 (每個約 5-6 分鐘)
# ================================================================

echo "========================================"
echo "  V8 BATCH TRAINING - All 20 Symbols"
echo "========================================"
echo ""
echo "開始時間: $(date)"
echo ""

# 拉取最新程式碼
git pull origin main

echo ""
echo "========== 開始訓練 =========="
echo ""

# 符號陣列
SYMBOLS=("BTC" "ETH" "BNB" "XRP" "ADA" "DOGE" "SOL" "DOT" "AVAX" "LINK" \
         "UNI" "LTC" "MATIC" "ARB" "OP" "ATOM" "FTM" "NEAR" "PEPE" "SHIB")

# 訓練計數器
COUNT=1
TOTAL=${#SYMBOLS[@]}

for SYMBOL in "${SYMBOLS[@]}"; do
    echo "[$COUNT/$TOTAL] Training $SYMBOL..."
    python training/train_lstm_v8_pure_regression.py --symbol "$SYMBOL" --epochs 150 --batch-size 16
    
    if [ $? -eq 0 ]; then
        echo "✓ $SYMBOL 訓練成功"
    else
        echo "✗ $SYMBOL 訓練失敗"
    fi
    
    echo ""
    COUNT=$((COUNT + 1))
done

echo "========================================"
echo "  訓練完成！"
echo "結束時間: $(date)"
echo "========================================"
echo ""
echo "查看結果:"
python visualize_results.py
