#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 訓練腳本 V7 - 混合模式 (修正版)

策略：
- 主任務：價格回歸 (V1 証實有效)
- 輔助任務：方向預測 (MSE loss, 不是第別)
- 沒有任務權重賽驟，所以不會丢失

為何 V7 會成功？
1. 價格予測是已穗詩群的目標
2. 方向是年外群
3. 兩者合作會蓮雲提低
用法:
  python training/train_lstm_v7_hybrid.py --symbol SOL
  python training/train_lstm_v7_hybrid.py --symbol BTC --direction-weight 0.3
"""

import os
import sys
import io
import json
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, accuracy_score

import ccxt
import logging

# 設定 Windows UTF-8 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CONFIG = {
    'model': {
        'type': 'lstm',
        'input_size': 44,
        'hidden_size': 128,  # 回歸層核
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
    },
    'training': {
        'epochs': 150,
        'batch_size': 8,  # 降低 batch size
        'learning_rate': 0.0001,  # 降低学习率
        'weight_decay': 0.00001,  # 降低設氣衷衰
        'lookback_window': 60,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'direction_weight': 0.2,  # 方向任務權重 (20% vs 80%)
    },
    'optimization': {
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'patience': 20,
        'min_delta': 1e-6,
        'grad_clip': 0.5,  # 梅度袪断
    },
    'data': {
        'timeframe': '1h',
        'limit': 1000,
        'normalize': True,
    },
    'device': 'cuda',
}


def setup_logging(symbol: str, version: str = 'v7'):
    """設定日誌"""
    global logger
    
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'train_lstm_{version}_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)


def fetch_data(symbol: str, timeframe: str = '1h', limit: int = 1000):
    """獲取加密貨幣數據"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"Fetching {limit} candles for {symbol}/{timeframe}...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ Fetched {len(df)} candles for {symbol}/{timeframe}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加技術指標"""
    try:
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_middle'] = sma20
        df['bb_lower'] = sma20 - (std20 * 2)
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # 動量
        df['momentum'] = df['close'].diff(10)
        
        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
        
        # 移動平均
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # 成交量指標
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        df = df.ffill()
        
        logger.info(f"✓ Added technical indicators")
        return df
    
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        raise


def prepare_sequences(X, y_price, y_direction, lookback=60):
    """整理序列"""
    X_seq, y_price_seq, y_dir_seq = [], [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_price_seq.append(y_price[i+lookback])
        y_dir_seq.append(y_direction[i+lookback])
    return np.array(X_seq), np.array(y_price_seq), np.array(y_dir_seq)


class HybridLSTM(nn.Module):
    """混合 LSTM 模型 - 主回歸 + 輔助方向"""
    
    def __init__(self, config):
        super(HybridLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional'],
            batch_first=True
        )
        
        lstm_output_size = config['model']['hidden_size'] * (2 if config['model']['bidirectional'] else 1)
        
        # 主任務：價格回歸
        self.price_head = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 價格預測
        )
        
        # 輔助任務：方向回歸
        self.direction_head = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']*0.5),
            nn.Linear(32, 1),
            nn.Tanh()  # 輸出 [-1, 1]
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        price = self.price_head(last_out)
        direction = self.direction_head(last_out)
        
        return price, direction


def train_epoch(model, train_loader, optimizer, device, config):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    nan_count = 0
    
    for X_batch, y_price, y_dir in train_loader:
        X_batch = X_batch.to(device).float()
        y_price = y_price.to(device).float().unsqueeze(-1)
        y_dir = y_dir.to(device).float().unsqueeze(-1)
        
        optimizer.zero_grad()
        
        price_pred, direction_pred = model(X_batch)
        
        # 主任務 (80%)
        price_loss = nn.MSELoss()(price_pred, y_price)
        # 輔助任務 (20%)
        direction_loss = nn.MSELoss()(direction_pred, y_dir)
        
        loss = price_loss * (1 - config['training']['direction_weight']) + \
                direction_loss * config['training']['direction_weight']
        
        # 抪斷 NaN
        if torch.isnan(loss):
            nan_count += 1
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimization']['grad_clip'])
        optimizer.step()
        
        total_loss += loss.item()
    
    if nan_count > 0:
        logger.warning(f"  Warning: {nan_count} NaN batches skipped")
    
    return total_loss / max(len(train_loader) - nan_count, 1)


def validate(model, val_loader, device, config):
    """驗證模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_price, y_dir in val_loader:
            X_batch = X_batch.to(device).float()
            y_price = y_price.to(device).float().unsqueeze(-1)
            y_dir = y_dir.to(device).float().unsqueeze(-1)
            
            price_pred, direction_pred = model(X_batch)
            
            price_loss = nn.MSELoss()(price_pred, y_price)
            direction_loss = nn.MSELoss()(direction_pred, y_dir)
            
            loss = price_loss * (1 - config['training']['direction_weight']) + \
                    direction_loss * config['training']['direction_weight']
            
            if not torch.isnan(loss):
                total_loss += loss.item()
    
    return total_loss / len(val_loader)


def cleanup_old_models(symbol: str):
    """清理叨篆的老模型"""
    current_path = f'models/saved/{symbol}_model.pth'
    if os.path.exists(current_path):
        os.makedirs('models/backup_v6', exist_ok=True)
        shutil.move(current_path, f'models/backup_v6/{symbol}_model_v6_old.pth')
        logger.info(f"✓ Cleaned up incompatible old model")


def main():
    global logger
    
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Training v7 - Hybrid (Price + Direction)')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--direction-weight', type=float, default=0.2, help='Direction task weight')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.symbol, version='v7')
    
    logger.info('='*80)
    logger.info('LSTM MODEL TRAINING (V7 - Hybrid Mode - FIXED)')
    logger.info('='*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Strategy: Price Regression (80%) + Direction Auxiliary (20%)")
    logger.info(f"Direction Weight: {args.direction_weight*100:.1f}%")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Batch Size: {args.batch_size}")
    
    # 加載配置
    config = DEFAULT_CONFIG.copy()
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['training']['batch_size'] = args.batch_size
    config['training']['direction_weight'] = args.direction_weight
    config['device'] = args.device
    
    # 清理叨篆的老模型
    cleanup_old_models(args.symbol)
    
    # 1. 獲取數據
    logger.info("\n[1/6] Fetching data...")
    df = fetch_data(args.symbol, timeframe='1h', limit=config['data']['limit'])
    
    # 2. 添加技術指標
    logger.info("[2/6] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. 特徵提取和方向標籤
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    # 方向標籤：正數 = 上漲, 負數 = 下跌
    y_direction = np.diff(y, prepend=0)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 確保 X 的特徵數為 44
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
    elif X_scaled.shape[1] < 44:
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    
    # 正規化方向標籤 ([-1, 1]) 並措推 NaN
    y_direction_normalized = np.clip(y_direction / (np.max(np.abs(y_direction)) + 1e-8), -1, 1)
    y_direction_normalized = np.nan_to_num(y_direction_normalized, 0.0)
    
    logger.info(f"✓ Feature matrix shape: {X_scaled.shape}")
    
    # 4. 準備序列
    logger.info("[3/6] Preparing sequences...")
    X_seq, y_seq, y_dir_seq = prepare_sequences(X_scaled, y_scaled, y_direction_normalized, config['training']['lookback_window'])
    
    # train/val/test 分割
    n_samples = len(X_seq)
    train_size = int(n_samples * config['training']['train_split'])
    val_size = int(n_samples * config['training']['val_split'])
    
    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]
    y_dir_train, y_dir_val, y_dir_test = y_dir_seq[:train_size], y_dir_seq[train_size:train_size+val_size], y_dir_seq[train_size+val_size:]
    
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. 建立 DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(y_dir_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val), torch.tensor(y_dir_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(y_dir_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 6. 建立模型
    logger.info("\n[4/6] Building model...")
    model = HybridLSTM(config)
    model.to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 7. 訓練
    logger.info("\n[5/6] Training...")
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=config['training']['epochs'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, config)
        val_loss = validate(model, val_loader, args.device, config)
        scheduler.step()
        
        if val_loss < best_val_loss - config['optimization']['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), f'models/saved/{args.symbol}_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 15 == 0 or patience_counter >= config['optimization']['patience']:
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if patience_counter >= config['optimization']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 8. 評估
    logger.info("\n" + "="*80)
    logger.info("EVALUATION - Hybrid Model")
    logger.info("="*80)
    
    model.load_state_dict(torch.load(f'models/saved/{args.symbol}_model.pth'))
    model.eval()
    
    with torch.no_grad():
        test_prices = []
        test_directions = []
        test_trues = []
        test_dir_trues = []
        
        for X_batch, y_batch, y_dir_batch in test_loader:
            X_batch = X_batch.to(args.device).float()
            price, direction = model(X_batch)
            
            test_prices.extend(price.cpu().numpy().flatten())
            test_directions.extend(direction.cpu().numpy().flatten())
            test_trues.extend(y_batch.numpy())
            test_dir_trues.extend(y_dir_batch.numpy())
    
    # 反正規化
    test_prices_inverse = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
    test_trues_inverse = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    # 計算指標
    mae = mean_absolute_error(test_trues_inverse, test_prices_inverse)
    mape = mean_absolute_percentage_error(test_trues_inverse, test_prices_inverse)
    rmse = np.sqrt(mean_squared_error(test_trues_inverse, test_prices_inverse))
    
    # 方向準確度
    dir_predictions = np.sign(np.array(test_directions))
    dir_trues = np.sign(np.array(test_dir_trues))
    dir_acc = accuracy_score(dir_trues, dir_predictions)
    
    logger.info(f"MAE:                {mae:.6f} USD")
    logger.info(f"MAPE:               {mape:.4f} %")
    logger.info(f"RMSE:               {rmse:.6f} USD")
    logger.info(f"Direction Accuracy: {dir_acc:.2%}")
    logger.info(f"Trainable Params:   {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("="*80)
    
    # 保存結果
    results = {
        'version': 'v7',
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'direction_accuracy': float(dir_acc),
        'test_samples': len(test_trues),
        'model_params': sum(p.numel() for p in model.parameters()),
        'direction_weight': args.direction_weight,
        'method': 'hybrid_regression_direction_auxiliary',
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.symbol}_results_v7.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/{args.symbol}_results_v7.json")


if __name__ == '__main__':
    main()
