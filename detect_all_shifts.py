#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detect All Symbol Shifts - Scan all models and generate bias correction config

Usage:
  python detect_all_shifts.py

Output:
  - models/bias_corrections_v8.json (配置文件)
  - shift_report.txt (詳細報告)
"""

import os
import sys
import io
import json
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
import logging

import ccxt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logging():
    global logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def get_available_models():
    """獲取所有可用模型"""
    saved_dir = 'models/saved'
    if not os.path.exists(saved_dir):
        logger.error(f"Error: Directory not found: {saved_dir}")
        return []
    
    models = []
    for file in os.listdir(saved_dir):
        if file.endswith('_model_v8.pth') or file.endswith('.pth'):
            symbol = file.replace('_model_v8.pth', '').replace('_model.pth', '').replace('.pth', '').upper()
            if symbol and symbol not in models:
                models.append(symbol)
    
    return sorted(list(set(models)))


def fetch_data(symbol: str, limit: int = 1000):
    """獲取最新數據"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        ohlcv = exchange.fetch_ohlcv(symbol_pair, '1h', limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    except Exception as e:
        logger.error(f"  Error fetching {symbol}: {e}")
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加44個技術指標"""
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
        
        df = df.ffill().bfill()
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        return df
    except Exception as e:
        logger.error(f"  Error adding indicators: {e}")
        return None


def detect_model_config(state_dict):
    """檢測模型架構"""
    try:
        weight_ih = state_dict.get('lstm.weight_ih_l0')
        if weight_ih is not None:
            hidden_size = weight_ih.shape[0] // 4
        else:
            hidden_size = 64
        
        bidirectional = 'lstm.weight_ih_l0_reverse' in state_dict
        
        num_layers = 1
        layer = 1
        while f'lstm.weight_ih_l{layer}' in state_dict:
            num_layers += 1
            layer += 1
        
        return {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'dropout': 0.3,
        }
    except:
        return {'hidden_size': 64, 'num_layers': 2, 'bidirectional': True, 'dropout': 0.3}


class RegressionLSTM(torch.nn.Module):
    """LSTM Model"""
    def __init__(self, input_size=44, hidden_size=64, num_layers=2, dropout=0.3, bidirectional=True):
        super(RegressionLSTM, self).__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(lstm_output_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        price = self.regressor(last_out)
        return price


def find_model_file(symbol: str):
    """找到模型文件"""
    saved_dir = 'models/saved'
    possible_names = [
        f'{symbol}_model_v8.pth',
        f'{symbol}_model.pth',
        f'{symbol}.pth',
    ]
    
    for name in possible_names:
        path = os.path.join(saved_dir, name)
        if os.path.exists(path):
            return path
    
    return None


def detect_symbol_shift(symbol: str):
    """檢測單個幣種的偏差"""
    logger.info(f"\n[Detecting] {symbol}...", extra={'symbol': symbol})
    
    try:
        # Fetch data
        df = fetch_data(symbol)
        if df is None or len(df) == 0:
            logger.warning(f"  Failed to fetch data")
            return None
        
        # Add indicators
        df = add_indicators(df)
        if df is None:
            logger.warning(f"  Failed to add indicators")
            return None
        
        # Extract features
        from sklearn.preprocessing import MinMaxScaler
        
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
        X = df[feature_cols].values
        y = df['close'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        if X_scaled.shape[1] > 44:
            X_scaled = X_scaled[:, :44]
        elif X_scaled.shape[1] < 44:
            padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
            X_scaled = np.hstack([X_scaled, padding])
        
        # Prepare sequences
        lookback = 60
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - lookback):
            X_seq.append(X_scaled[i:i+lookback])
            y_seq.append(y_scaled[i+lookback])
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # Split
        n = len(X_seq)
        train_end = int(n * 0.8)
        val_end = train_end + int(n * 0.1)
        
        X_test = X_seq[val_end:]
        y_test = y_seq[val_end:]
        
        # Load model
        model_path = find_model_file(symbol)
        if not model_path:
            logger.warning(f"  Model not found")
            return None
        
        state_dict = torch.load(model_path, map_location=device)
        config = detect_model_config(state_dict)
        
        model = RegressionLSTM(
            input_size=44,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Predict
        with torch.no_grad():
            test_prices = []
            for i in range(0, len(X_test), 32):
                X_batch = torch.tensor(X_test[i:i+32], dtype=torch.float32).to(device)
                price = model(X_batch)
                test_prices.extend(price.cpu().numpy().flatten())
        
        # Inverse transform
        test_prices_real = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
        test_real = scaler_y.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()
        
        # Calculate shift
        mean_shift = test_prices_real.mean() - test_real.mean()
        
        logger.info(f"  Mean Shift: {mean_shift:.8f}")
        
        return {
            'symbol': symbol,
            'mean_shift': float(mean_shift),
            'actual_mean': float(test_real.mean()),
            'predicted_mean': float(test_prices_real.mean()),
            'mae': float(np.abs(test_prices_real - test_real).mean()),
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"  Error: {e}")
        return None


def main():
    setup_logging()
    
    logger.info('\n' + '='*60)
    logger.info('Detecting All Symbol Shifts - V8 Model')
    logger.info('='*60)
    
    # Get models
    symbols = get_available_models()
    if not symbols:
        logger.error("No models found")
        return
    
    logger.info(f"\nFound {len(symbols)} models: {', '.join(symbols)}")
    logger.info(f"\nScanning...\n")
    
    # Detect shifts
    results = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol}...")
        result = detect_symbol_shift(symbol)
        if result:
            results.append(result)
        else:
            logger.warning(f"  Skipped")
    
    if not results:
        logger.error("No successful detections")
        return
    
    # Generate config
    logger.info(f"\n\n{'='*60}")
    logger.info(f"Generating bias correction config...")
    logger.info(f"{'='*60}\n")
    
    bias_config = {
        "version": "v8",
        "description": "Bias correction offsets for each cryptocurrency model",
        "generated_at": pd.Timestamp.now().isoformat(),
        "corrections": {}
    }
    
    # Build corrections
    for result in results:
        symbol = result['symbol']
        shift = result['mean_shift']
        bias_config['corrections'][symbol] = round(shift, 8)
        
        logger.info(f"{symbol:8} -> Correction: {shift:+.8f}")
    
    # Save config
    os.makedirs('models', exist_ok=True)
    config_path = 'models/bias_corrections_v8.json'
    
    with open(config_path, 'w') as f:
        json.dump(bias_config, f, indent=2)
    
    logger.info(f"\nSaved: {config_path}")
    
    # Generate report
    logger.info(f"\nGenerating report...")
    
    report = f"""# V8 Model Bias Correction Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total Models Scanned: {len(results)}

## Bias Corrections (Add to prediction to correct)

| Symbol | Correction | Actual Mean | Predicted Mean | MAE |
|--------|-----------|-------------|----------------|-----|
"""
    
    for result in sorted(results, key=lambda x: abs(x['mean_shift']), reverse=True):
        symbol = result['symbol']
        shift = result['mean_shift']
        actual = result['actual_mean']
        predicted = result['predicted_mean']
        mae = result['mae']
        
        report += f"| {symbol:8} | {shift:+.8f} | {actual:15.8f} | {predicted:16.8f} | {mae:10.8f} |\n"
    
    report += f"""\n## Python Usage

```python
import json

with open('models/bias_corrections_v8.json', 'r') as f:
    bias_config = json.load(f)

def correct_prediction(symbol, predicted_price):
    correction = bias_config['corrections'].get(symbol, 0)
    return predicted_price + correction

# Usage
raw_prediction = model_output  # e.g., 3148.0
corrected = correct_prediction('ETH', raw_prediction)  # 3191.0
```

## Model Details

"""
    
    for result in results:
        report += f"\n### {result['symbol']}\n"
        report += f"- Correction: {result['mean_shift']:+.8f}\n"
        report += f"- MAE: {result['mae']:.8f}\n"
    
    report_path = 'shift_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved: {report_path}")
    
    logger.info(f"\n\n{'='*60}")
    logger.info(f"Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"\nGenerated files:")
    logger.info(f"  - {config_path}")
    logger.info(f"  - {report_path}")
    logger.info(f"\nUse corrections in your models and trading bot!")


if __name__ == '__main__':
    main()
