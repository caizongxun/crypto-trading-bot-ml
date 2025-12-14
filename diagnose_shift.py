#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose Prediction Shift - Check if shift is visualization issue or model error

Usage:
  python diagnose_shift.py --symbol PEPE
"""

import os
import sys
import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import ccxt
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logging():
    global logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def fetch_data(symbol: str, limit: int = 1000):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"Fetching {limit} candles for {symbol}...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, '1h', limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def diagnose(symbol: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"Diagnosing Prediction Shift - {symbol}")
    logger.info(f"{'='*60}")
    
    # Fetch and prepare data
    df = fetch_data(symbol)
    if df is None:
        return
    
    df = add_indicators(df)
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
    test_end = n
    
    X_test = X_seq[val_end:test_end]
    y_test = y_seq[val_end:test_end]
    
    # Create dummy model (just predict mean)
    y_test_scaled = y_test
    y_test_real = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Statistics
    logger.info(f"\nData Statistics:")
    logger.info(f"  Original close price range: [{y.min():.8f}, {y.max():.8f}]")
    logger.info(f"  Training set range: [{y[:train_end+lookback].min():.8f}, {y[:train_end+lookback].max():.8f}]")
    logger.info(f"  Test set range: [{y[val_end+lookback:].min():.8f}, {y[val_end+lookback:].max():.8f}]")
    logger.info(f"  Scaler range: [{scaler_y.data_min_[0]:.8f}, {scaler_y.data_max_[0]:.8f}]")
    
    logger.info(f"\nTest Set Analysis:")
    logger.info(f"  Test set size: {len(y_test_real)}")
    logger.info(f"  Test set mean: {y_test_real.mean():.8f}")
    logger.info(f"  Test set std: {y_test_real.std():.8f}")
    logger.info(f"  Test set min: {y_test_real.min():.8f}")
    logger.info(f"  Test set max: {y_test_real.max():.8f}")
    
    logger.info(f"\nScaler Analysis:")
    logger.info(f"  Train vs Test: Mean difference = {y[:train_end+lookback].mean() - y_test_real.mean():.8f}")
    
    # Check if test set is below or above training range
    train_min = y[:train_end+lookback].min()
    train_max = y[:train_end+lookback].max()
    test_min = y_test_real.min()
    test_max = y_test_real.max()
    
    logger.info(f"\nDistribution Check:")
    if test_min < train_min and test_max < train_max:
        logger.info(f"  WARNING: Test set is LOWER than training set")
        logger.info(f"  This causes models to predict LOW values")
        shift_reason = "Lower Test Range"
    elif test_min > train_min and test_max > train_max:
        logger.info(f"  WARNING: Test set is HIGHER than training set")
        logger.info(f"  This causes models to predict HIGH values")
        shift_reason = "Higher Test Range"
    else:
        logger.info(f"  OK: Test set is within training range")
        shift_reason = "Distribution OK"
    
    # Try loading and testing with actual model
    model_path = f'models/saved/{symbol}_model_v8.pth'
    if os.path.exists(model_path):
        logger.info(f"\nLoading actual model: {model_path}")
        
        from visualize_all_v8 import RegressionLSTM, detect_model_config
        
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
        
        with torch.no_grad():
            test_prices = []
            for i in range(0, len(X_test), 32):
                X_batch = torch.tensor(X_test[i:i+32], dtype=torch.float32).to(device)
                price = model(X_batch)
                test_prices.extend(price.cpu().numpy().flatten())
        
        test_prices_real = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_real, test_prices_real)
        mape = mean_absolute_percentage_error(y_test_real, test_prices_real)
        
        logger.info(f"\nModel Prediction Analysis:")
        logger.info(f"  Predicted mean: {test_prices_real.mean():.8f}")
        logger.info(f"  Predicted std: {test_prices_real.std():.8f}")
        logger.info(f"  Predicted min: {test_prices_real.min():.8f}")
        logger.info(f"  Predicted max: {test_prices_real.max():.8f}")
        logger.info(f"\n  MAE: {mae:.8f}")
        logger.info(f"  MAPE: {mape:.6f}%")
        
        # Check prediction shift
        mean_shift = test_prices_real.mean() - y_test_real.mean()
        logger.info(f"\nPrediction Shift Analysis:")
        logger.info(f"  Mean shift: {mean_shift:.8f}")
        if abs(mean_shift) < mae:
            logger.info(f"  Shift < MAE: Shift is NORMAL (within model error)")
        else:
            logger.info(f"  Shift >= MAE: Shift indicates BIAS")
        
        # Generate comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Predictions
        ax = axes[0, 0]
        x = np.arange(len(y_test_real))
        ax.plot(x, y_test_real, 'b-', label='Actual', linewidth=2, alpha=0.7)
        ax.plot(x, test_prices_real, 'r-', label='Predicted', linewidth=2, alpha=0.7)
        ax.set_title(f'{symbol} Predictions vs Actual')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error
        ax = axes[0, 1]
        errors = test_prices_real - y_test_real
        ax.plot(errors, 'g-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=mean_shift, color='r', linestyle='--', label=f'Mean shift: {mean_shift:.6f}')
        ax.set_title('Prediction Error (Pred - Actual)')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Distribution
        ax = axes[1, 0]
        ax.hist(y_test_real, bins=20, alpha=0.5, label='Actual', color='blue')
        ax.hist(test_prices_real, bins=20, alpha=0.5, label='Predicted', color='red')
        ax.set_title('Price Distribution')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Metrics
        ax = axes[1, 1]
        ax.axis('off')
        metrics_text = f"""
        Shift Diagnosis: {shift_reason}
        
        MAE: {mae:.8f}
        MAPE: {mape:.6f}%
        
        Mean Shift: {mean_shift:.8f}
        
        Training Range: [{train_min:.8f}, {train_max:.8f}]
        Test Range: [{test_min:.8f}, {test_max:.8f}]
        Pred Range: [{test_prices_real.min():.8f}, {test_prices_real.max():.8f}]
        
        Interpretation:
        {'OK' if mae > 1e-5 else 'SHIFT IS NORMAL'} - Within model error range
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_shift_diagnosis.png', dpi=120, bbox_inches='tight')
        logger.info(f"\nDiagnosis plot saved: {symbol}_shift_diagnosis.png")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Conclusion: {shift_reason}")
    logger.info(f"{'='*60}")


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Diagnose prediction shift')
    parser.add_argument('--symbol', type=str, required=True, help='Crypto symbol')
    args = parser.parse_args()
    
    diagnose(args.symbol.upper())


if __name__ == '__main__':
    main()
