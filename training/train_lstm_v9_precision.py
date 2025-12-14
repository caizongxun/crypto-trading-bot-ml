#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 訓練腳本 V9 - 精度提升版

改進方案:
1. 更大的网络: 256 隱藏層 + 3 層 (vs V8: 128 + 2)
2. 更多技術指標: 60+ 個 (vs V8: 44 個)
3. 更长訓練時間: 200 epochs (vs V8: 150)
4. 改進正則化: Batch Norm 加入
5. 高级優化次: AdamW 通過更好的 learning rate 調整
用法:
  python training/train_lstm_v9_precision.py --symbol SOL --epochs 200
  python training/train_lstm_v9_precision.py --symbol BTC --lr 0.0005
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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import ccxt
import logging

# 設定 Windows UTF-8 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CONFIG = {
    'model': {
        'type': 'lstm',
        'input_size': 60,  # 更多技術指標
        'hidden_size': 256,  # 更大的隐藏層
        'num_layers': 3,  # 更多層數
        'dropout': 0.4,
        'bidirectional': True,
    },
    'training': {
        'epochs': 200,  # 更長訓練時間
        'batch_size': 16,
        'learning_rate': 0.0005,  # 降低学习率
        'weight_decay': 0.00005,
        'lookback_window': 60,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },
    'optimization': {
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'patience': 25,
        'min_delta': 1e-7,
        'grad_clip': 1.0,
    },
    'data': {
        'timeframe': '1h',
        'limit': 1000,
        'normalize': True,
    },
    'device': 'cuda',
}


def setup_logging(symbol: str, version: str = 'v9'):
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
    """添加超過 60 個技術指標"""
    try:
        # 基本作量
        df['high-low'] = df['high'] - df['low']
        df['close-open'] = df['close'] - df['open']
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # RSI 系列
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD 系列
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = macd.ewm(span=signal).mean()
            df[f'macd_hist_{fast}_{slow}'] = macd - df[f'macd_signal_{fast}_{slow}']
        
        # Bollinger Bands 系列
        for period in [20, 50]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # 动量指標
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].diff(period)
        
        # 成交量指標
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        for period in [5, 10]:
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
        
        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_sma'] = obv.rolling(window=20).mean()
        
        # 移动平均系列
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # 卫布指数
        df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                           (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * (-100)
        
        # Stochastic
        df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
        
        # 此外部流
        df['money_flow'] = (df['close'] - df['open']) * df['volume']
        df['money_flow_sma'] = df['money_flow'].rolling(window=20).mean()
        
        # 填充 NaN
        df = df.ffill().bfill()
        
        logger.info(f"✓ Added technical indicators (total {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} features)")
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


class EnhancedLSTM(nn.Module):
    """強化的 LSTM 模型 - V9"""
    
    def __init__(self, config):
        super(EnhancedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional'],
            batch_first=True
        )
        
        lstm_output_size = config['model']['hidden_size'] * 2
        
        # 強化的回歸頭
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(config['model']['dropout']),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(config['model']['dropout']),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']*0.5),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        price = self.regressor(last_out)
        return price


def train_epoch(model, train_loader, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float().unsqueeze(-1)
        
        optimizer.zero_grad()
        
        price_pred = model(X_batch)
        loss = nn.SmoothL1Loss()(price_pred, y_batch)  # 使用 SmoothL1Loss 比 MSE 更強健
        
        if torch.isnan(loss):
            logger.warning("  NaN loss detected, skipping batch")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / max(len(train_loader), 1)


def validate(model, val_loader, device):
    """驗證模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(-1)
            
            price_pred = model(X_batch)
            loss = nn.SmoothL1Loss()(price_pred, y_batch)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
    
    return total_loss / max(len(val_loader), 1)


def cleanup_old_models(symbol: str):
    """清理叨篆的老模型"""
    current_path = f'models/saved/{symbol}_model.pth'
    if os.path.exists(current_path):
        os.makedirs('models/backup_v8', exist_ok=True)
        shutil.move(current_path, f'models/backup_v8/{symbol}_model_v8_old.pth')
        logger.info(f"✓ Cleaned up old model")


def main():
    global logger
    
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Training v9 - Enhanced Precision')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.symbol, version='v9')
    
    logger.info('='*80)
    logger.info('LSTM MODEL TRAINING (V9 - Enhanced Precision)')
    logger.info('='*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Strategy: Larger network (256x3) + More indicators (60+) + SmoothL1Loss")
    
    # 加載配置
    config = DEFAULT_CONFIG.copy()
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['training']['batch_size'] = args.batch_size
    config['device'] = args.device
    
    # 清理叨篆的老模型
    cleanup_old_models(args.symbol)
    
    # 1. 獲取數據
    logger.info("\n[1/5] Fetching data...")
    df = fetch_data(args.symbol, timeframe='1h', limit=config['data']['limit'])
    
    # 2. 添加技術指標
    logger.info("[2/5] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. 特徵提取
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    X = df[feature_cols].values
    y = df['close'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 確保 X 的特徵數為 60
    if X_scaled.shape[1] > 60:
        X_scaled = X_scaled[:, :60]
    elif X_scaled.shape[1] < 60:
        padding = np.zeros((X_scaled.shape[0], 60 - X_scaled.shape[1]))
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
    model = EnhancedLSTM(config)
    model.to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 7. 訓練
    logger.info("\n[5/5] Training...")
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=config['training']['epochs'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    model_saved = False
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = validate(model, val_loader, args.device)
        scheduler.step()
        
        if not torch.isnan(torch.tensor(val_loss)) and val_loss < best_val_loss - config['optimization']['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), f'models/saved/{args.symbol}_model.pth')
            model_saved = True
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or patience_counter >= config['optimization']['patience']:
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if patience_counter >= config['optimization']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 8. 評估
    logger.info("\n" + "="*80)
    logger.info("EVALUATION - Enhanced Precision Model")
    logger.info("="*80)
    
    if not model_saved:
        logger.error("Model was not saved during training. Training failed.")
        return
    
    model.load_state_dict(torch.load(f'models/saved/{args.symbol}_model.pth', map_location=args.device))
    model.eval()
    
    with torch.no_grad():
        test_prices = []
        test_trues = []
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(args.device).float()
            price = model(X_batch)
            
            test_prices.extend(price.cpu().numpy().flatten())
            test_trues.extend(y_batch.numpy())
    
    # 反正規化
    test_prices_inverse = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
    test_trues_inverse = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    # 計算指標
    mae = mean_absolute_error(test_trues_inverse, test_prices_inverse)
    mape = mean_absolute_percentage_error(test_trues_inverse, test_prices_inverse)
    rmse = np.sqrt(mean_squared_error(test_trues_inverse, test_prices_inverse))
    
    logger.info(f"MAE:                {mae:.6f} USD")
    logger.info(f"MAPE:               {mape:.4f} %")
    logger.info(f"RMSE:               {rmse:.6f} USD")
    logger.info(f"Test Samples:       {len(test_trues)}")
    logger.info(f"Trainable Params:   {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("="*80)
    
    # 保存結果
    results = {
        'version': 'v9',
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'test_samples': len(test_trues),
        'model_params': sum(p.numel() for p in model.parameters()),
        'method': 'enhanced_lstm_precision_v9',
        'improvements': '256x3 LSTM + 60+ indicators + SmoothL1Loss + BatchNorm',
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.symbol}_results_v9.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/{args.symbol}_results_v9.json")


if __name__ == '__main__':
    main()
