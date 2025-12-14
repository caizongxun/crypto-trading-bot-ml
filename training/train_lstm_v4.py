#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 訓練腳本 v4 - LoRA 子訓練版本
使用 LoRA (Low-Rank Adaptation) 技費專注於方向預測
輙透模型，添加方向標簽輸出

特色：
- 基於 v1 模型的 LoRA 子訓練
- 三個輸出肠
  1. 價格預測
  2. 方向標簽
  3. 幅度預測
- 支持回滾到 v1/v2/v3

用法:
  python training/train_lstm_v4.py --symbol SOL
  python training/train_lstm_v4.py --symbol BTC --epochs 150
  python training/train_lstm_v4.py --symbol ETH --lora-rank 8
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
        'output_size': 1,  # 價格預測
        'dropout': 0.3,
        'bidirectional': True,
    },
    'training': {
        'epochs': 150,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'lookback_window': 60,
        'forecast_horizon': 1,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
    },
    'lora': {
        'rank': 4,  # LoRA 秘秘雨
        'alpha': 8,  # LoRA 画幅
        'dropout': 0.1,
    },
    'optimization': {
        'optimizer': 'adamw',
        'scheduler': 'cosine',
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


def setup_logging(symbol: str, version: str = 'v4'):
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
        
        df = df.fillna(method='bfill')
        
        logger.info(f"✓ Added technical indicators")
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


class LoRALayer(nn.Module):
    """LoRA 適配層"""
    
    def __init__(self, in_features, out_features, rank=4, alpha=8, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 權重
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LSTMWithLoRA(nn.Module):
    """带 LoRA 的 LSTM 模型"""
    
    def __init__(self, config):
        super(LSTMWithLoRA, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional'],
            batch_first=True
        )
        
        lstm_output_size = config['model']['hidden_size'] * (2 if config['model']['bidirectional'] else 1)
        
        # 標沖線性層
        self.fc_base = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
        )
        
        # 價格預測輸出
        self.price_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 價格
        )
        
        # 方向預測輸出 (有 LoRA)
        self.direction_lora = LoRALayer(32, 16, 
                                        rank=config['lora']['rank'],
                                        alpha=config['lora']['alpha'],
                                        dropout=config['lora']['dropout'])
        self.direction_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # 輸出 [-1, 1]
        )
        
        # 幅度預測輸出 (有 LoRA)
        self.volatility_lora = LoRALayer(32, 16,
                                         rank=config['lora']['rank'],
                                         alpha=config['lora']['alpha'],
                                         dropout=config['lora']['dropout'])
        self.volatility_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # 輸出 > 0
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        # 基標特徵提取
        base_features = self.fc_base(last_out)
        
        # 三個輸出
        price = self.price_head(base_features)
        
        direction_lora = self.direction_lora(base_features)
        direction = self.direction_head(direction_lora)
        
        volatility_lora = self.volatility_lora(base_features)
        volatility = self.volatility_head(volatility_lora)
        
        return price, direction, volatility


def load_model_from_v1(symbol: str, config: dict) -> LSTMWithLoRA:
    """從 v1 模型加載並轉換結構"""
    v1_model_path = f'models/saved/{symbol}_model.pth'
    
    if not os.path.exists(v1_model_path):
        logger.warning(f"v1 model not found: {v1_model_path}")
        return None
    
    try:
        # 先建立 v4 模型
        model = LSTMWithLoRA(config)
        
        # 加載 v1 模型改篇
        v1_state = torch.load(v1_model_path, map_location=device)
        
        # 笼藏地加載可相容的怹數
        model_dict = model.state_dict()
        v1_compatible = {k: v for k, v in v1_state.items() if k in model_dict}
        model_dict.update(v1_compatible)
        model.load_state_dict(model_dict, strict=False)
        
        logger.info(f"✓ Loaded v1 model and converted: {v1_model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load v1 model: {e}")
        return None


def backup_models(symbol: str):
    """備份前鼠版本模型"""
    current_path = f'models/saved/{symbol}_model.pth'
    
    if os.path.exists(current_path):
        os.makedirs('models/backup_v3', exist_ok=True)
        shutil.copy(current_path, f'models/backup_v3/{symbol}_model_v3.pth')
        logger.info(f"✓ v3 model backed up")


def train_epoch(model, train_loader, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch, y_dir in train_loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).float().unsqueeze(-1)
        y_dir = y_dir.to(device).float().unsqueeze(-1)
        
        optimizer.zero_grad()
        
        price, direction, volatility = model(X_batch)
        
        # 多任務名意損失
        price_loss = nn.MSELoss()(price, y_batch)
        direction_loss = nn.MSELoss()(direction, y_dir)  # 方向標簽
        volatility_loss = 0.1 * nn.L1Loss()(volatility, torch.ones_like(volatility))  # 輕權重
        
        loss = price_loss + 0.5 * direction_loss + volatility_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    """驗證模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch, y_dir in val_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float().unsqueeze(-1)
            y_dir = y_dir.to(device).float().unsqueeze(-1)
            
            price, direction, volatility = model(X_batch)
            
            price_loss = nn.MSELoss()(price, y_batch)
            direction_loss = nn.MSELoss()(direction, y_dir)
            volatility_loss = 0.1 * nn.L1Loss()(volatility, torch.ones_like(volatility))
            
            loss = price_loss + 0.5 * direction_loss + volatility_loss
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def calculate_direction_accuracy(y_true, y_pred):
    """計算方向準確度"""
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    return accuracy_score(y_true_dir, y_pred_dir)


def main():
    global logger
    
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Training v4 - LoRA with Direction Label')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--lora-rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.symbol, version='v4')
    
    logger.info('='*80)
    logger.info('LSTM MODEL TRAINING (V4 - LoRA with Direction Label)')
    logger.info('='*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Device: {args.device}")
    logger.info(f"LoRA Rank: {args.lora_rank}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"\nMulti-task Learning:")
    logger.info(f"  - Price Prediction")
    logger.info(f"  - Direction Label (with LoRA)")
    logger.info(f"  - Volatility Prediction (with LoRA)")
    
    # 加載配置
    config = DEFAULT_CONFIG.copy()
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['lora']['rank'] = args.lora_rank
    config['device'] = args.device
    
    # 備份模型
    backup_models(args.symbol)
    
    # 1. 獲取數據
    logger.info("\n[1/6] Fetching data...")
    df = fetch_data(args.symbol, timeframe='1h', limit=config['data']['limit'])
    
    # 2. 添加技術指標
    logger.info("[2/6] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. 特徵提取和正規化
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    # 方向標簽：-1 (下跌) 或 1 (上漲)
    y_direction = np.diff(y, prepend=0)
    y_direction = np.sign(y_direction)
    
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
    logger.info("[3/6] Preparing sequences...")
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, config['training']['lookback_window'])
    _, y_dir_seq = prepare_sequences(X_scaled, y_direction, config['training']['lookback_window'])
    
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
    model = load_model_from_v1(args.symbol, config)
    
    if model is None:
        logger.info("Creating new model...")
        model = LSTMWithLoRA(config)
    
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
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = validate(model, val_loader, args.device)
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
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
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
        test_prices = []
        test_directions = []
        test_trues = []
        test_dir_trues = []
        
        for X_batch, y_batch, y_dir_batch in test_loader:
            X_batch = X_batch.to(args.device).float()
            price, direction, volatility = model(X_batch)
            
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
    dir_acc = calculate_direction_accuracy(np.array(test_dir_trues), np.array(test_directions))
    
    logger.info(f"MAE:                {mae:.6f} USD")
    logger.info(f"MAPE:               {mape:.4f} %")
    logger.info(f"RMSE:               {rmse:.6f} USD")
    logger.info(f"Direction Accuracy: {dir_acc:.2%}")
    logger.info(f"Trainable Params:   {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("="*80)
    
    # 保存結果
    results = {
        'version': 'v4',
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'direction_accuracy': float(dir_acc),
        'test_samples': len(test_trues),
        'model_params': sum(p.numel() for p in model.parameters()),
        'lora_rank': args.lora_rank,
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.symbol}_results_v4.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/{args.symbol}_results_v4.json")


if __name__ == '__main__':
    main()
