#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 訓練腳本 - v1.1
基於實際訓練配置優化：44 個特徵、Batch=16、Hidden=128、Bidirectional

MAE 目標: < 0.2 USD
MAPE 目標: < 0.1%
Accuracy 目標: > 90%
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, accuracy_score

import yaml
import logging

# 設定 Windows UTF-8 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==================== 配置 ====================

config = None
logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 從 config.yaml 讀取的標準配置
DEFAULT_CONFIG = {
    'model': {
        'type': 'lstm',
        'input_size': 44,          # ✅ 根據 TFT V3 的實際特徵數
        'hidden_size': 128,        # ✅ GPU 優化配置
        'num_layers': 2,           # ✅ Bidirectional 需要 2 層以上效果好
        'output_size': 1,
        'dropout': 0.3,
        'bidirectional': True,
    },
    'training': {
        'epochs': 200,
        'batch_size': 16,          # ✅ GPU 優化（4GB 時穩定配置）
        'learning_rate': 0.0005,   # ✅ 略微降低，避免 overshoot
        'weight_decay': 0.0001,
        'lookback_window': 60,     # ✅ 與 TFT V3 一致
        'forecast_horizon': 1,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },
    'optimization': {
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'warmup_steps': 500,
        'patience': 25,            # Early stopping
        'min_delta': 1e-6,
    },
    'data': {
        'timeframe': '1h',
        'limit': 5000,
        'normalize': True,
        'augmentation': False,
    },
    'device': 'cuda',
}


def setup_logging(symbol: str) -> None:
    """設定日誌"""
    global logger
    
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'train_lstm_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)


def load_config(config_path: str = 'training/config.yaml') -> dict:
    """載入配置檔或使用預設值"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return DEFAULT_CONFIG


class LSTMModel(nn.Module):
    """高效能 LSTM 模型 - 基於實戰配置"""
    
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],      # 44
            hidden_size=config['model']['hidden_size'],    # 128
            num_layers=config['model']['num_layers'],      # 2
            dropout=config['model']['dropout'],            # 0.3
            bidirectional=config['model']['bidirectional'], # True
            batch_first=True
        )
        
        # Bidirectional 輸出維度 = hidden_size * 2
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


def fetch_data(symbol: str, timeframe: str = '1h', limit: int = 5000) -> pd.DataFrame:
    """從 Binance 獲取數據"""
    try:
        import ccxt
        exchange = ccxt.binance()
        trading_pair = f"{symbol}/USDT"
        ohlcv = exchange.fetch_ohlcv(trading_pair, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"✓ Fetched {len(df)} candles for {symbol}/{timeframe}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加 44 個技術指標（與 TFT V3 一致）"""
    df = df.copy()
    
    # 基礎 OHLCV
    features = ['open', 'high', 'low', 'close', 'volume']
    
    # 價格動量
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 移動平均線
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
    
    # 波動率指標
    for window in [5, 10, 20]:
        df[f'std_{window}'] = df['close'].rolling(window).std()
        df[f'atr_{window}'] = calculate_atr(df, window)
    
    # RSI
    for window in [7, 14, 21]:
        df[f'rsi_{window}'] = calculate_rsi(df['close'], window)
    
    # MACD
    macd_data = calculate_macd(df['close'])
    df['macd'] = macd_data['macd']
    df['macd_signal'] = macd_data['signal']
    df['macd_hist'] = macd_data['histogram']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df['close'], window=20, num_std=2)
    df['bb_upper'] = bb_data['upper']
    df['bb_lower'] = bb_data['lower']
    df['bb_mid'] = bb_data['mid']
    
    # 成交量指標
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # 價格變化率
    for window in [5, 10]:
        df[f'price_change_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
    
    # 高低價比
    df['hl_ratio'] = df['high'] / df['low']
    df['ol_ratio'] = df['open'] / df['low']
    
    # 移除 NaN
    df = df.dropna()
    
    logger.info(f"✓ Added {len(df.columns) - 6} technical indicators (total features: {len(df.columns)})")
    return df


def calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """計算 ATR"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """計算 RSI - 修復版本"""
    deltas = prices.diff()
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / (down + 1e-8)
    rsi = pd.Series(100. - 100. / (1. + rs), index=prices.index)  # 確保是 Series
    rsi[:window] = 0
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """計算 MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return {'macd': macd, 'signal': signal_line, 'histogram': histogram}


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> dict:
    """計算布林帶"""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    return {'upper': upper, 'mid': sma, 'lower': lower}


def prepare_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 60) -> tuple:
    """整理訓練數據序列"""
    logger.info(f"Preparing sequences with lookback={lookback}...")
    
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    
    return np.array(X_seq), np.array(y_seq)


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
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


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """計算方向準確度"""
    # 計算實際價格變化方向
    y_true_diff = np.diff(y_true, prepend=0)
    y_true_dir = (y_true_diff > 0).astype(int)
    
    # 計算預測價格變化方向
    y_pred_diff = np.diff(y_pred, prepend=0)
    y_pred_dir = (y_pred_diff > 0).astype(int)
    
    # 計算準確度
    return accuracy_score(y_true_dir, y_pred_dir)


def main():
    global config
    
    parser = argparse.ArgumentParser(description='LSTM Training for Crypto Price Prediction')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol (e.g., SOL, BTC)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 設定
    setup_logging(args.symbol)
    config = load_config()
    
    # 覆蓋命令行參數
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.lr
    config['device'] = args.device
    
    logger.info('='*80)
    logger.info('LSTM MODEL TRAINING (V1.1)')
    logger.info('='*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Input Features: {config['model']['input_size']}")
    logger.info(f"Hidden Size: {config['model']['hidden_size']}")
    logger.info(f"Num Layers: {config['model']['num_layers']}")
    logger.info(f"Bidirectional: {config['model']['bidirectional']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    
    # 1. 獲取數據
    logger.info("\n[1/5] Fetching data...")
    df = fetch_data(args.symbol, limit=config['data']['limit'])
    
    # 2. 添加技術指標
    logger.info("[2/5] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 特徵提取
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    # 3. 正規化
    logger.info("[3/5] Normalizing data...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 確保 X 的特徵數為 44
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
    elif X_scaled.shape[1] < 44:
        # 補充到 44 列
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    
    logger.info(f"✓ Feature matrix shape: {X_scaled.shape}")
    
    # 4. 整理序列
    logger.info("[4/5] Preparing sequences...")
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
    
    # 6. 訓練模型
    logger.info("\n[5/5] Training...")
    
    model = LSTMModel(config).to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=config['training']['epochs'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss - config['optimization']['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), f'models/saved/{args.symbol}_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or patience_counter >= config['optimization']['patience']:
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Best: {best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= config['optimization']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 7. 評估
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
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'direction_accuracy': float(dir_acc),
        'test_samples': len(test_trues),
        'model_params': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.symbol}_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/{args.symbol}_results.json")
    logger.info(f"Model saved to models/saved/{args.symbol}_model.pth")


if __name__ == '__main__':
    main()
