#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V8 模型訓練腳本 - 支持所有幣種

使用方法:
  python train_v8_models.py              # 訓練所有幣種
  python train_v8_models.py --symbol SOL # 訓練單個幣種
  python train_v8_models.py --symbol BTC,ETH,SOL # 訓練多個幣種

模型配置 (V8):
  - 輸入特徵: 44 個技術指標
  - 隱藏層: 128 x 2
  - Bidirectional LSTM
  - Dropout: 0.3
  - 訓練 Epochs: 150
  - Batch Size: 32
  - Early Stopping: True
"""

import os
import sys
import io
import argparse
from pathlib import Path
from datetime import datetime

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

# V8 配置 (44 個技術指標)
MODEL_CONFIG = {
    'input_size': 44,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
    'lookback': 60,
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
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


def fetch_training_data(symbol: str, timeframe: str = '1h', limit: int = 2000):
    """接取訓練數據"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"  📊 接取 {limit} 根蠟燭 {symbol}/{timeframe}...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"  ✓ 接取完成 {len(df)} 根蠟燭")
        return df
    
    except Exception as e:
        logger.error(f"  ✗ 接取數據失敗: {e}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加 44 個技術指標 (V8 版本)"""
    try:
        logger.info(f"  📈 添加技術指標...")
        
        # 基本作用
        df['high-low'] = df['high'] - df['low']
        df['close-open'] = df['close'] - df['open']
        df['returns'] = df['close'].pct_change()
        
        # RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
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
        
        # 計算實際特徵數
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
        logger.info(f"  ✓ 添加了 {len(feature_cols)} 個技術指標")
        
        return df
    
    except Exception as e:
        logger.error(f"  ✗ 添加技術指標失敗: {e}")
        return None


def prepare_sequences(X, y, lookback=60):
    """準備序列數據"""
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class RegressionLSTM(nn.Module):
    """V8 LSTM 模型"""
    
    def __init__(self, input_size=44, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True):
        super(RegressionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
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
    """提前停止機制"""
    
    def __init__(self, patience=20, min_delta=0.0):
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


def train_model(symbol: str):
    """訓練 V8 模型"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 開始訓練 {symbol} V8 模型")
    logger.info(f"{'='*60}")
    
    # 接取數據
    df = fetch_training_data(symbol)
    if df is None or len(df) == 0:
        logger.error(f"  ✗ 接取 {symbol} 數據失敗")
        return False
    
    # 添加技術指標
    df = add_technical_indicators(df)
    if df is None:
        logger.error(f"  ✗ 添加技術指標失敗")
        return False
    
    # 特徵提取
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    logger.info(f"  📦 原始特徵: {X.shape}")
    
    # 標準化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 確保特徵數為 44 (V8 標準)
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
        logger.info(f"  ✓ 特徵數超過 44，已截斷")
    elif X_scaled.shape[1] < 44:
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
        logger.info(f"  ✓ 特徵數不足 44，已用零填充")
    
    logger.info(f"  ✓ 標準化後特徵: {X_scaled.shape}")
    
    # 準備序列
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, MODEL_CONFIG['lookback'])
    logger.info(f"  ✓ 序列數據: {X_seq.shape}")
    
    # 分割數據
    n_samples = len(X_seq)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    logger.info(f"  ✓ 數據分割: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 創建 DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'])
    
    # 創建模型
    model = RegressionLSTM(
        input_size=MODEL_CONFIG['input_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout'],
        bidirectional=MODEL_CONFIG['bidirectional']
    )
    model.to(device)
    
    logger.info(f"  ✓ 模型創建完成")
    logger.info(f"     - 隱藏層: {MODEL_CONFIG['hidden_size']} x {MODEL_CONFIG['num_layers']}")
    logger.info(f"     - Bidirectional: {MODEL_CONFIG['bidirectional']}")
    logger.info(f"     - Dropout: {MODEL_CONFIG['dropout']}")
    
    # 損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG['learning_rate'],
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    # 提前停止
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
    
    # 訓練
    logger.info(f"\n  📚 開始訓練 {MODEL_CONFIG['epochs']} epochs...\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, MODEL_CONFIG['epochs'] + 1):
        # 訓練
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
        scheduler.step(val_loss)
        
        # 打印進度
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{MODEL_CONFIG['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_dir = Path('models/saved')
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f'{symbol}_model_v8.pth'
            torch.save(model.state_dict(), str(model_path))
        
        # 提前停止
        if early_stopping(val_loss):
            logger.info(f"\n  ⏹️ 提前停止於 Epoch {epoch}")
            break
    
    # 測試
    logger.info(f"\n  🧪 測試模型...")
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
    
    logger.info(f"\n  📊 測試結果:")
    logger.info(f"     MAE:  {mae:.6f} USD")
    logger.info(f"     MAPE: {mape:.4f} %")
    logger.info(f"     RMSE: {rmse:.6f} USD")
    
    # 保存最終模型
    model_dir = Path('models/saved')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{symbol}_model_v8.pth'
    torch.save(model.state_dict(), str(model_path))
    
    logger.info(f"\n  ✓ 模型已保存: {model_path}")
    logger.info(f"{'='*60}\n")
    
    return True


def get_available_symbols():
    """獲取常見幣種列表"""
    return [
        'BTC', 'ETH', 'ADA', 'DOGE', 'SOL', 'XRP', 'LINK', 'ATOM',
        'AVAX', 'FTM', 'NEAR', 'MATIC', 'ARB', 'OP', 'LTC', 'DOT',
        'LTCBTC', 'BNB', 'LTC'
    ]


def main():
    global logger
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='V8 模型訓練腳本')
    parser.add_argument('--symbol', type=str, default=None, help='幣種符號 (逗號分隔)')
    args = parser.parse_args()
    
    logger.info('\n' + '='*60)
    logger.info('V8 模型訓練腳本')
    logger.info('='*60)
    logger.info(f"\n💻 設備: {device}")
    logger.info(f"📦 配置: 44 特徵 | 128x2 隱藏層 | 150 Epochs")
    
    # 決定要訓練的幣種
    if args.symbol:
        symbols = [s.upper().strip() for s in args.symbol.split(',')]
    else:
        symbols = get_available_symbols()
    
    logger.info(f"\n🎯 要訓練的幣種: {', '.join(symbols)}\n")
    
    # 訓練每個幣種
    success_count = 0
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] 訓練 {symbol}...\n")
        if train_model(symbol):
            success_count += 1
        logger.info("\n" + "="*60)
    
    # 完成摘要
    logger.info(f"\n✅ 完成訓練")
    logger.info(f"{'='*60}")
    logger.info(f"\n✓ 成功訓練: {success_count}/{len(symbols)} 個幣種")
    logger.info(f"\n📁 模型保存位置: models/saved/")
    logger.info(f"\n💡 接下來可以運行: python visualize_all_v8.py")


if __name__ == '__main__':
    main()
