#!/bin/bash

# ============================================================================
# 訓練市值前 20 的幣種 - 一个一个訓練 (不用 YAML)
# ============================================================================
# 用途：顺序訓練每个幣種
# 平台：Linux / macOS
# ============================================================================

set -e

echo ""
echo "================================================================================"
echo "加密貨幣 LSTM 模型 - 訓練市值前 20 的幣種"
echo "================================================================================"
echo ""

# 設置訓練參数
EPOCHS=200
BATCH_SIZE=16
DEVICE=cuda

echo "訓練配置："
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""
echo "正在訓練市值前 20 的幣種..."
echo ""

# 初始化计数器
COUNT=0
SUCCESS=0
FAILED=0

# 訓練每个幣種
for symbol in BTC ETH BNB SOL XRP DOGE ADA AVAX LINK MATIC ARB OP LDO SUI NEAR INJ SEI TON FET ICP; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/20] 訓練 $symbol..."
    
    if python training/train_lstm_v1.py --symbol "$symbol" --epochs $EPOCHS --device $DEVICE; then
        SUCCESS=$((SUCCESS + 1))
        echo "  ✓ $symbol 訓練成功"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ $symbol 訓練失敗"
    fi
    echo ""
done

echo "================================================================================"
echo "訓練完成！总结回请"
echo "================================================================================"
echo "  成功: $SUCCESS/20"
echo "  失敗: $FAILED/20"
echo ""
echo "檔索位置："
echo "  - models/saved/"
echo "  - results/"
echo ""
