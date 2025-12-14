#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 訓練腳本 v5 - 方向分類版本
使用方向分類代替佳棰，不是回歸也不是多任務

特色：
- 上漢/下跌二分類（使用 CrossEntropyLoss）
- 強力的上漢辨識力
- 簡配可聖体的一直一核架構（沒有趨児的配置）

用法:
  python training/train_lstm_v5.py --symbol SOL
  python training/train_lstm_v5.py --symbol BTC --epochs 200
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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

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
        'output_size': 2,  # 2 分類
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


def setup_logging(symbol: str, version: str = 'v5'):
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


def prepare_sequences(X, y_labels, lookback=60):
    """整理序列"""
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y_labels[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class DirectionLSTM(nn.Module):
    """方向分類 LSTM 模型"""
    
    def __init__(self, config):
        super(DirectionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional'],
            batch_first=True
        )
        
        lstm_output_size = config['model']['hidden_size'] * (2 if config['model']['bidirectional'] else 1)
        
        # 分類頭
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(32, config['model']['output_size'])  # 2 類（上漢/下跌）
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits


def load_model_from_v1(symbol: str, config: dict) -> DirectionLSTM:
    """從 v1 模型加載並轉換"""
    v1_model_path = f'models/saved/{symbol}_model.pth'
    
    if not os.path.exists(v1_model_path):
        logger.warning(f"v1 model not found: {v1_model_path}")
        return None
    
    try:
        model = DirectionLSTM(config)
        v1_state = torch.load(v1_model_path, map_location=device)
        
        # 笼藏地加載可相容的怹數
        model_dict = model.state_dict()
        v1_compatible = {k: v for k, v in v1_state.items() if k in model_dict and v1_state[k].shape == model_dict[k].shape}
        model_dict.update(v1_compatible)
        model.load_state_dict(model_dict, strict=False)
        
        logger.info(f"✓ Loaded v1 model and converted to v5")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load v1 model: {e}")
        return None


def train_epoch(model, train_loader, optimizer, device, class_weights=None):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device, class_weights=None):
    """驗證模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).long()
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(val_loader), accuracy


def evaluate_direction(y_true, y_pred):
    """計算方向準確度和其他指標"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def backup_models(symbol: str):
    """備份前鼠版本模型"""
    current_path = f'models/saved/{symbol}_model.pth'
    
    if os.path.exists(current_path):
        os.makedirs('models/backup_v4', exist_ok=True)
        shutil.copy(current_path, f'models/backup_v4/{symbol}_model_v4.pth')
        logger.info(f"✓ v4 model backed up")


def main():
    global logger
    
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Training v5 - Direction Classification')
    parser.add_argument('--symbol', type=str, default='SOL', help='Trading symbol')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.symbol, version='v5')
    
    logger.info('='*80)
    logger.info('LSTM MODEL TRAINING (V5 - Direction Classification)')
    logger.info('='*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Method: 2-class direction classification (0=down, 1=up)")
    
    # 加載配置
    config = DEFAULT_CONFIG.copy()
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['device'] = args.device
    
    # 備份模型
    backup_models(args.symbol)
    
    # 1. 獲取數據
    logger.info("\n[1/6] Fetching data...")
    df = fetch_data(args.symbol, timeframe='1h', limit=config['data']['limit'])
    
    # 2. 添加技術指標
    logger.info("[2/6] Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # 3. 特徵提取和方向標簽
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    # 方向標簽：0 (下跌) 或 1 (上漢)
    y_direction = np.diff(y, prepend=0)
    y_labels = (y_direction > 0).astype(int)  # ́ 0 or 1
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 確保 X 的特徵數為 44
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
    elif X_scaled.shape[1] < 44:
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    
    logger.info(f"✓ Feature matrix shape: {X_scaled.shape}")
    
    # 計算類別權重（幣例失衡）
    class_counts = np.bincount(y_labels)
    class_weights = torch.tensor([1.0 / (c + 1e-6) for c in class_counts], dtype=torch.float32).to(args.device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    logger.info(f"Class distribution: {class_counts[0]} down, {class_counts[1]} up")
    logger.info(f"Class weights: {class_weights}")
    
    # 4. 準備序列
    logger.info("[3/6] Preparing sequences...")
    X_seq, y_seq = prepare_sequences(X_scaled, y_labels, config['training']['lookback_window'])
    
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
    logger.info("\n[4/6] Building model...")
    model = load_model_from_v1(args.symbol, config)
    
    if model is None:
        logger.info("Creating new model...")
        model = DirectionLSTM(config)
    
    model.to(args.device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 7. 訓練
    logger.info("\n[5/6] Training...")
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=config['training']['epochs'])
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, class_weights)
        val_loss, val_acc = validate(model, val_loader, args.device, class_weights)
        scheduler.step()
        
        if val_acc > best_val_acc + config['optimization']['min_delta']:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs('models/saved', exist_ok=True)
            torch.save(model.state_dict(), f'models/saved/{args.symbol}_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or patience_counter >= config['optimization']['patience']:
            logger.info(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}")
        
        if patience_counter >= config['optimization']['patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 8. 評估
    logger.info("\n" + "="*80)
    logger.info("EVALUATION - Direction Classification Metrics")
    logger.info("="*80)
    
    model.load_state_dict(torch.load(f'models/saved/{args.symbol}_model.pth'))
    model.eval()
    
    with torch.no_grad():
        test_preds = []
        test_labels = []
        
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(args.device).float()
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y_batch.numpy())
    
    # 計算指標
    accuracy, precision, recall, f1 = evaluate_direction(test_labels, test_preds)
    
    logger.info(f"Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Precision:      {precision:.4f}")
    logger.info(f"Recall:         {recall:.4f}")
    logger.info(f"F1 Score:       {f1:.4f}")
    logger.info(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("="*80)
    
    # 保存結果
    results = {
        'version': 'v5',
        'symbol': args.symbol,
        'timestamp': datetime.now().isoformat(),
        'direction_accuracy': float(accuracy),
        'direction_precision': float(precision),
        'direction_recall': float(recall),
        'direction_f1': float(f1),
        'test_samples': len(test_labels),
        'model_params': sum(p.numel() for p in model.parameters()),
        'method': 'binary_classification',
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.symbol}_results_v5.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to results/{args.symbol}_results_v5.json")


if __name__ == '__main__':
    main()
