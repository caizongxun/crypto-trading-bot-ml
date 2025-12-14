#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 訓練腳本 v2 - 方向準確度優化版本
基於 v1 模型進行微調，增強方向預測準確度

特色：
- 從 v1 模型加載並繼續訓練
- 專注於方向準確度優化
- 支援回滾到 v1

用法:
  python training/train_lstm_v2.py --symbol SOL
  python training/train_lstm_v2.py --symbol BTC --epochs 100
  python training/train_lstm_v2.py --symbol ETH --load-v1
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
        'hidden_size': 128,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.3,
        'bidirectional': True,
    },
    'training': {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.0001,  # v2: 較低的學習率用於微調
        'weight_decay': 0.0001,
        'lookback_window': 60,
        'forecast_horizon': 1,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },
    'optimization': {
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_steps': 100,
        'patience': 20,
        'min_delta': 1e-6,
    },
    'data': {
        'timeframe': '1h',
        'limit': 1000,
        'normalize': True,
    },
    'device': 'cuda',
}


def setup_logging(symbol: str, version: str = 'v2'):
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
        
        # ADX
        df['plus_dm'] = df['high'].diff()
        df['minus_dm'] = -df['low'].diff()
        df['plus_dm'] = df['plus_dm'].where(df['plus_dm'] > 0, 0)
        df['minus_dm'] = df['minus_dm'].where(df['minus_dm'] > 0, 0)
        
        # 移動平均
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        # 成交量指標
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        df = df.fillna(method='bfill')
        
        logger.info(f"✓ Added technical indicators (total features: {len(df.columns) - 2})")
        return df
    
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        raise


def prepare_sequences(X, y, lookback=60):
    """整理序列"""
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class LSTMModel(nn.Module):
    """高效能 LSTM 模型"""
    
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional'],
            batch_first=True
        )
        
        lstm_output_size = config['model']['hidden_size'] * (2 if config['model']['bidirectional'] else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(32, config['model']['output_size'])
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


def load_model_from_v1(symbol: str, config: dict) -> LSTMModel:
    """從 v1 加載模型"""
    v1_model_path = f'models/saved/{symbol}_model.pth'
    
    if not os.path.exists(v1_model_path):
        logger.warning(f"v1 model not found: {v1_model_path}, creating new model")
        return None
    
    try:
        model = LSTMModel(config)
        model.load_state_dict(torch.load(v1_model_path, map_location=device))
        logger.info(f"✓ Loaded v1 model: {v1_model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load v1 model: {e}")
        return None


def backup_v1_model(symbol: str):
    """備份 v1 模型"""
    v1_path = f'models/saved/{symbol}_model.pth'
    backup_path = f'models/backup_v1/{symbol}_model_v1.pth'
    
    if os.path.exists(v1_path):
        os.makedirs('models/backup_v1', exist_ok=True)
        shutil.copy(v1_path, backup_path)
        logger.info(f"✓ v1 model backed up: {backup_path}")


def train_epoch(model, train_loader, optimizer, criterion, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float().unsqueeze(-1)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """驗證模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(-1)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def calculate_direction_accuracy(y_true, y_pred):
    """計算方向準確度"""
    y_true_diff = np.diff(y_true, prepend=0)
    y_true_dir = (y_true_diff > 0).astype(int)
    
    y_pred_diff = np.diff(y_pred, prepend=0)
    y_pred_dir = (y_pred_diff > 0).astype(int)
    
    return accuracy_score(y_true_dir, y_pred_dir)


def main():
    global logger
    
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Training v2 - Direction Accuracy Optimization')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--load-v1', action='store_true', help='Load from v1 model')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.symbol, version='v2')
    
    logger.info('='*80)
    logger.info('LSTM MODEL TRAINING (V2 - Direction Accuracy Optimization)')
    logger.info('='*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Load from v1: {args.load_v1}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    
    # 加載配置
    config = DEFAULT_CONFIG.copy()
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['device'] = args.device
    
    # 備份 v1 模型
    backup_v1_model(args.symbol)
    
    # 1. 獲取數據
    logger.info("\n[1/5] Fetching data...")
    df = fetch_data(args.symbol, timeframe='1h', limit=config['data']['limit'])
    
    # 2. 添加技術指標
    logger.info("[2/5] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. 特徵提取和正規化
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
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
    
    logger.info(f"✓ Feature matrix shape: {X_scaled.shape}")
    
    # 4. 準備序列
    logger.info("[3/5] Preparing sequences...")
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, config['training']['lookback_window'])
    
    # train/val/test 分割
    n_samples = len(X_seq)
    train_size = int(n_samples * config['training']['train_split'])
    val_size = int(n_samples * config['training']['val_split'])
    
    X_train, X_val, X_test = X_seq[:train_size], X_seq[train_size:train_size+val_size], X_seq[train_size+val_size:]
    y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]
    
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 5. 建立 DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 6. 建立模型
    logger.info("\n[4/5] Building model...")
    model = None
    
    if args.load_v1:
        model = load_model_from_v1(args.symbol, config)
    
    if model is None:
        logger.info("Creating new model...")
        model = LSTMModel(config)
    
    model.to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 7. 訓練
    logger.info("\n[5/5] Training...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=config['training']['epochs'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)
        scheduler.step()
        
        if val_loss < best_val_loss - config['optimization']['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), f'models/saved/{args.symbol}_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or patience_counter >= config['optimization']['patience']:
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
        
        if patience_counter >= config['optimization']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 8. 評估
    logger.info("\n" + "="*80)
    logger.info("EVALUATION")
    logger.info("="*80)
    
    model.load_state_dict(torch.load(f'models/saved/{args.symbol}_model.pth'))
    model.eval()
    
    with torch.no_grad():
        test_preds = []
        test_trues = []
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(args.device).float()
            outputs = model(X_batch)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_trues.extend(y_batch.numpy())
    
    # 反正規化
    test_preds_inverse = scaler_y.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
    test_trues_inverse = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    # 計算指標
    mae = mean_absolute_error(test_trues_inverse, test_preds_inverse)
    mape = mean_absolute_percentage_error(test_trues_inverse, test_preds_inverse)
    rmse = np.sqrt(mean_squared_error(test_trues_inverse, test_preds_inverse))
    dir_acc = calculate_direction_accuracy(test_trues_inverse, test_preds_inverse)
    
    logger.info(f"MAE:                {mae:.6f} USD")
    logger.info(f"MAPE:               {mape:.4f} %")
    logger.info(f"RMSE:               {rmse:.6f} USD")
    logger.info(f"Direction Accuracy: {dir_acc:.2%}")
    logger.info("="*80)
    
    # 保存結果
    results = {
        'version': 'v2',
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'direction_accuracy': float(dir_acc),
        'test_samples': len(test_trues),
        'model_params': sum(p.numel() for p in model.parameters()),
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.symbol}_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/{args.symbol}_results_v2.json")
    logger.info(f"Model saved to models/saved/{args.symbol}_model.pth")
    logger.info(f"Backup of v1 model saved to models/backup_v1/{args.symbol}_model_v1.pth")


if __name__ == '__main__':
    main()
