#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超參數調整工具 - 自動找出最佳訓練配置

用法:
  python hyperparameter_tuning.py --symbol SOL                    # 核心調優
  python hyperparameter_tuning.py --symbol SOL --fast             # 快速模式 (模範減少)
  python hyperparameter_tuning.py --symbol SOL --comprehensive    # 全面模式 (嘗試所有組合)

調整範疇:
  - hidden_size: [64, 128, 256]
  - num_layers: [1, 2, 3]
  - dropout: [0.1, 0.3, 0.5]
  - learning_rate: [0.0001, 0.001, 0.01]
  - batch_size: [16, 32, 64]
"""

import os
import sys
import io
import argparse
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import ccxt
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 基础配置
BASE_CONFIG = {
    'input_size': 44,
    'lookback': 60,
    'epochs': 100,  # 調整時用較短的 epochs
    'early_stop_patience': 15,
}


def setup_logging():
    global logger
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)


def fetch_training_data(symbol: str, limit: int = 1500):
    """接取訓練數據"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"    📊 接取 {limit} 根蠟燭...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, '1h', limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        logger.error(f"    ✗ 接取數據失敗: {e}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加 44 個技術指標"""
    try:
        df['high-low'] = df['high'] - df['low']
        df['close-open'] = df['close'] - df['open']
        df['returns'] = df['close'].pct_change()
        
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_middle'] = sma20
        df['bb_lower'] = sma20 - (std20 * 2)
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        df['momentum'] = df['close'].diff(10)
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
        
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        df = df.ffill()
        
        return df
    
    except Exception as e:
        logger.error(f"    ✗ 添加技術指標失敗: {e}")
        return None


def prepare_sequences(X, y, lookback=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class RegressionLSTM(nn.Module):
    def __init__(self, input_size=44, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True):
        super(RegressionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        price = self.regressor(last_out)
        return price


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


def train_and_evaluate(symbol: str, config: dict, data: tuple) -> dict:
    """訓練並評估模型"""
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = data
    
    # 創建 DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # 創建模型
    model = RegressionLSTM(
        input_size=BASE_CONFIG['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=True
    )
    model.to(device)
    
    # 損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    
    early_stopping = EarlyStopping(patience=BASE_CONFIG['early_stop_patience'])
    
    # 訓練
    best_val_loss = float('inf')
    for epoch in range(BASE_CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if early_stopping(val_loss):
            break
    
    # 測試
    model.eval()
    with torch.no_grad():
        test_prices = []
        test_trues = []
        
        for i in range(0, len(X_test), 32):
            X_batch = torch.tensor(X_test[i:i+32], dtype=torch.float32).to(device)
            price = model(X_batch)
            test_prices.extend(price.cpu().numpy().flatten())
            test_trues.extend(y_test[i:i+32])
    
    # 反標準化
    test_prices_inverse = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
    test_trues_inverse = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    # 計算指標
    mae = mean_absolute_error(test_trues_inverse, test_prices_inverse)
    mape = mean_absolute_percentage_error(test_trues_inverse, test_prices_inverse)
    rmse = np.sqrt(mean_squared_error(test_trues_inverse, test_prices_inverse))
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'val_loss': best_val_loss,
    }


def prepare_data(symbol: str):
    """準備訓練數據"""
    logger.info(f"\n  📐 準備数据...")
    
    df = fetch_training_data(symbol)
    if df is None:
        return None
    
    df = add_technical_indicators(df)
    if df is None:
        return None
    
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 确保特征数为 44
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
    elif X_scaled.shape[1] < 44:
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, BASE_CONFIG['lookback'])
    
    n_samples = len(X_seq)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler_y)


def hyperparameter_tuning(symbol: str, mode='core'):
    """超參數調整"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 超參數調整 - {symbol} ({mode.upper()})")
    logger.info(f"{'='*60}")
    
    # 準備數據
    data = prepare_data(symbol)
    if data is None:
        logger.error("\n  ✗ 數據準備失敗")
        return None
    
    # 定義测試空間
    if mode == 'fast':
        # 快速模式 - 简少的組合
        hidden_sizes = [128]
        num_layers_list = [2]
        dropouts = [0.3]
        learning_rates = [0.001]
        batch_sizes = [32]
    
    elif mode == 'core':
        # 核心模式 - 擇選最可能的組合
        hidden_sizes = [64, 128, 256]
        num_layers_list = [1, 2, 3]
        dropouts = [0.1, 0.3, 0.5]
        learning_rates = [0.0001, 0.001, 0.01]
        batch_sizes = [16, 32, 64]
    
    else:  # comprehensive
        # 全面模式 - 所有組合
        hidden_sizes = [64, 128, 192, 256]
        num_layers_list = [1, 2, 3, 4]
        dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
        learning_rates = [0.00001, 0.0001, 0.001, 0.01]
        batch_sizes = [8, 16, 32, 64, 128]
    
    # 估算總數
    total_combinations = len(hidden_sizes) * len(num_layers_list) * len(dropouts) * len(learning_rates) * len(batch_sizes)
    logger.info(f"\n  📋 測試組合: {total_combinations} 個\n")
    
    results = []
    count = 0
    
    for config_tuple in product(hidden_sizes, num_layers_list, dropouts, learning_rates, batch_sizes):
        count += 1
        hidden_size, num_layers, dropout, lr, batch_size = config_tuple
        
        config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': lr,
            'batch_size': batch_size,
        }
        
        logger.info(f"  [{count}/{total_combinations}] HS={hidden_size} | NL={num_layers} | DO={dropout} | LR={lr} | BS={batch_size}")
        
        try:
            result = train_and_evaluate(symbol, config, data)
            result.update(config)
            results.append(result)
            
            logger.info(f"         MAE={result['mae']:.6f} | MAPE={result['mape']:.4f}% | RMSE={result['rmse']:.6f}")
        
        except Exception as e:
            logger.warning(f"         ✗ 訓練失敗: {e}")
            continue
    
    if not results:
        logger.error("\n  ✗ 估伐失敗")
        return None
    
    # 排序結果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mae').reset_index(drop=True)
    
    # 輸出最佳配置
    logger.info(f"\n\n{'='*60}")
    logger.info(f"🌟 最佳配置 (Top 5)")
    logger.info(f"{'='*60}\n")
    
    for i, row in results_df.head(5).iterrows():
        logger.info(f"\n#{i+1} - MAE: {row['mae']:.6f}")
        logger.info(f"  hidden_size={int(row['hidden_size'])} | num_layers={int(row['num_layers'])}")
        logger.info(f"  dropout={row['dropout']:.1f} | learning_rate={row['learning_rate']}")
        logger.info(f"  batch_size={int(row['batch_size'])}")
        logger.info(f"  MAPE={row['mape']:.4f}% | RMSE={row['rmse']:.6f}")
    
    # 保存結果
    results_file = f'tuning_results_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"\n\n✓ 結果已保存: {results_file}")
    
    # 輸出最佳配置 JSON
    best_config = results_df.iloc[0].to_dict()
    config_file = f'best_config_{symbol}.json'
    
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2, default=str)
    
    logger.info(f"✓ 最佳配置已保存: {config_file}")
    logger.info(f"\n{repr(best_config)}")
    
    return best_config


def main():
    global logger
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='超參數調整工具')
    parser.add_argument('--symbol', type=str, required=True, help='幣種符號')
    parser.add_argument('--fast', action='store_true', help='快速模式')
    parser.add_argument('--comprehensive', action='store_true', help='全面模式')
    args = parser.parse_args()
    
    mode = 'fast' if args.fast else ('comprehensive' if args.comprehensive else 'core')
    
    logger.info('\n' + '='*60)
    logger.info('📋 超參數調整工具')
    logger.info('='*60)
    logger.info(f"\n💻 設備: {device}")
    logger.info(f"👀 模式: {mode.upper()}")
    
    hyperparameter_tuning(args.symbol.upper(), mode)


if __name__ == '__main__':
    main()
