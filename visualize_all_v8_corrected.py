#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
All Cryptocurrency V8 Visualization Tool - With Bias Correction

Automatically applies bias corrections from models/bias_corrections_v8.json

Usage:
  python visualize_all_v8_corrected.py              # All symbols
  python visualize_all_v8_corrected.py --symbol SOL # Single symbol
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime
from glob import glob

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

MODEL_CONFIG = {
    'input_size': 44,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
}

# Global bias corrections
BIAS_CORRECTIONS = {}


def setup_logging():
    global logger
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)


def load_bias_corrections():
    """載入偏差校正配置"""
    global BIAS_CORRECTIONS
    
    config_path = 'models/bias_corrections_v8.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                bias_config = json.load(f)
            BIAS_CORRECTIONS = bias_config.get('corrections', {})
            logger.info(f"Loaded bias corrections for {len(BIAS_CORRECTIONS)} symbols")
            return True
        except Exception as e:
            logger.warning(f"Could not load bias corrections: {e}")
            return False
    else:
        logger.warning(f"Bias corrections file not found: {config_path}")
        logger.warning(f"Run 'python detect_all_shifts.py' first to generate it")
        return False


def apply_correction(symbol: str, predicted_price):
    """應用偏差校正"""
    correction = BIAS_CORRECTIONS.get(symbol, 0)
    corrected = predicted_price + correction
    return corrected, correction


def get_available_models():
    saved_dir = 'models/saved'
    if not os.path.exists(saved_dir):
        logger.error(f"Error: Directory not found: {saved_dir}")
        return []
    
    models = []
    for file in os.listdir(saved_dir):
        if file.endswith('_model_v8.pth') or file.endswith('.pth'):
            symbol = file.replace('_model_v8.pth', '').replace('_model.pth', '').replace('.pth', '').upper()
            if symbol:
                models.append(symbol)
    
    return sorted(list(set(models)))


def detect_model_config(state_dict):
    """Detect model architecture from saved weights"""
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
        return MODEL_CONFIG.copy()


def fetch_data(symbol: str, timeframe: str = '1h', limit: int = 1000):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"    Fetching {limit} candles {symbol}/{timeframe}...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"    Got {len(df)} candles")
        return df
    
    except Exception as e:
        logger.error(f"    Error fetching data: {e}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add 44 technical indicators"""
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
        logger.error(f"    Error adding indicators: {e}")
        return None


def prepare_sequences(X, y, lookback=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class RegressionLSTM(torch.nn.Module):
    """V8 LSTM Model with flexible architecture"""
    
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


def predict_symbol(symbol: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {symbol} (V8 Corrected)...")
    logger.info(f"{'='*60}")
    
    # Fetch data
    df = fetch_data(symbol)
    if df is None or len(df) == 0:
        logger.error(f"  Error: Failed to fetch {symbol} data")
        return None
    
    # Add indicators
    df = add_technical_indicators(df)
    if df is None:
        logger.error(f"  Error: Failed to add indicators")
        return None
    
    # Extract features
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
    elif X_scaled.shape[1] < 44:
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    
    logger.info(f"  Feature shape (V8): {X_scaled.shape}")
    
    # Prepare sequences
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, 60)
    
    # Split data
    n_samples = len(X_seq)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    logger.info(f"  Data split: Train={train_size}, Val={val_size}, Test={len(X_test)}")
    
    # Find and load model
    model_path = find_model_file(symbol)
    if not model_path:
        logger.error(f"  Error: Model not found for {symbol}")
        return None
    
    # Load weights to detect architecture
    state_dict = torch.load(model_path, map_location=device)
    detected_config = detect_model_config(state_dict)
    
    logger.info(f"  Detected config: hidden_size={detected_config['hidden_size']}, num_layers={detected_config['num_layers']}")
    
    # Create model with detected architecture
    model = RegressionLSTM(
        input_size=44,
        hidden_size=detected_config['hidden_size'],
        num_layers=detected_config['num_layers'],
        dropout=detected_config['dropout'],
        bidirectional=detected_config['bidirectional']
    )
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"  Model loaded successfully")
    
    # Predict
    with torch.no_grad():
        test_prices = []
        test_trues = []
        
        for i in range(0, len(X_test), 32):
            X_batch = torch.tensor(X_test[i:i+32], dtype=torch.float32).to(device)
            price = model(X_batch)
            test_prices.extend(price.cpu().numpy().flatten())
            test_trues.extend(y_test[i:i+32])
    
    # Inverse transform
    test_prices_inverse = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
    test_trues_inverse = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    # Apply correction
    test_prices_corrected, correction = apply_correction(symbol, test_prices_inverse)
    if isinstance(test_prices_corrected, np.ndarray):
        test_prices_corrected = test_prices_corrected
    else:
        test_prices_corrected = test_prices_inverse + correction
    
    # Calculate metrics
    mae_raw = mean_absolute_error(test_trues_inverse, test_prices_inverse)
    mae_corrected = mean_absolute_error(test_trues_inverse, test_prices_corrected)
    mape_raw = mean_absolute_percentage_error(test_trues_inverse, test_prices_inverse)
    mape_corrected = mean_absolute_percentage_error(test_trues_inverse, test_prices_corrected)
    
    logger.info(f"\n  Raw Predictions:")
    logger.info(f"    MAE:  {mae_raw:.8f}")
    logger.info(f"    MAPE: {mape_raw:.6f}%")
    
    logger.info(f"\n  After Correction (+{correction:.8f}):")
    logger.info(f"    MAE:  {mae_corrected:.8f}")
    logger.info(f"    MAPE: {mape_corrected:.6f}%")
    
    return {
        'symbol': symbol,
        'predicted_raw': test_prices_inverse,
        'predicted_corrected': test_prices_corrected,
        'actual': test_trues_inverse,
        'correction': correction,
        'mae_raw': mae_raw,
        'mae_corrected': mae_corrected,
        'mape_raw': mape_raw,
        'mape_corrected': mape_corrected,
    }


def main():
    global logger
    
    setup_logging()
    
    # Load bias corrections first
    corrections_loaded = load_bias_corrections()
    
    parser = argparse.ArgumentParser(description='V8 All Cryptocurrency Visualization Tool (With Bias Correction)')
    parser.add_argument('--symbol', type=str, default=None, help='Crypto symbols (comma-separated)')
    args = parser.parse_args()
    
    logger.info('\n' + '='*60)
    logger.info('V8 All Cryptocurrency Visualization Tool - CORRECTED')
    logger.info('='*60)
    logger.info(f"Bias corrections loaded: {corrections_loaded}")
    
    available_models = get_available_models()
    if not available_models:
        logger.error("\nError: No models found")
        return
    
    logger.info(f"\nFound {len(available_models)} models: {', '.join(available_models)}")
    
    if args.symbol:
        symbols = [s.upper().strip() for s in args.symbol.split(',')]
        symbols = [s for s in symbols if s in available_models]
        if not symbols:
            logger.error("Error: None of the specified symbols have models")
            return
    else:
        symbols = available_models
    
    logger.info(f"\nProcessing {len(symbols)} symbols...\n")
    
    results = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...\n")
        result = predict_symbol(symbol)
        if result:
            results.append(result)
            
            logger.info(f"\n  Generating visualization...")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Raw predictions
            ax = axes[0]
            x = np.arange(len(result['actual']))
            ax.plot(x, result['actual'], 'b-', label='Actual Price', linewidth=2, alpha=0.8)
            ax.plot(x, result['predicted_raw'], 'r--', label='Raw Prediction', linewidth=1.5, alpha=0.6)
            ax.plot(x, result['predicted_corrected'], 'g-', label='Corrected Prediction', linewidth=2, alpha=0.8)
            ax.set_title(f"{symbol} V8 Raw vs Corrected Predictions", fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=10)
            ax.set_ylabel('Price', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Metrics comparison
            ax = axes[1]
            ax.axis('off')
            metrics_text = f"""
{symbol} Performance Comparison

Bias Correction: {result['correction']:+.8f}

Raw Predictions:
  MAE:  {result['mae_raw']:.8f}
  MAPE: {result['mape_raw']:.6f}%

Corrected Predictions:
  MAE:  {result['mae_corrected']:.8f}
  MAPE: {result['mape_corrected']:.6f}%

Improvement:
  MAE Reduction:  {(result['mae_raw'] - result['mae_corrected']):.8f}
  MAPE Reduction: {(result['mape_raw'] - result['mape_corrected']):.6f}%
            """
            ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(f'{symbol}_predictions_v8_corrected.png', dpi=120, bbox_inches='tight')
            logger.info(f"    Saved: {symbol}_predictions_v8_corrected.png")
            plt.close()
        else:
            logger.warning(f"  Error: Failed to process {symbol}, skipping")
    
    if not results:
        logger.error("\nError: No successful predictions")
        return
    
    logger.info(f"\n\n{'='*60}")
    logger.info(f"Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"\nGenerated files:")
    for result in results:
        logger.info(f"  - {result['symbol']}_predictions_v8_corrected.png")
    logger.info(f"\nAll predictions are now bias-corrected! ✓")


if __name__ == '__main__':
    main()
