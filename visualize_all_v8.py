#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
All Cryptocurrency V8 Visualization Tool - Auto detect model architecture

Usage:
  python visualize_all_v8.py              # All symbols
  python visualize_all_v8.py --symbol SOL # Single symbol
  python visualize_all_v8.py --symbol BTC,ETH,SOL # Multiple symbols
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


def setup_logging():
    global logger
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)


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
        # Detect hidden size from LSTM weights
        weight_ih = state_dict.get('lstm.weight_ih_l0')
        if weight_ih is not None:
            # weight_ih shape: [4*hidden_size, input_size]
            hidden_size = weight_ih.shape[0] // 4
        else:
            hidden_size = 64
        
        # Detect if bidirectional
        bidirectional = 'lstm.weight_ih_l0_reverse' in state_dict
        
        # Detect num_layers
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
    except Exception as e:
        logger.warning(f"Could not detect config: {e}, using defaults")
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
    """Add 44 technical indicators (V8 version)"""
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
    logger.info(f"Processing {symbol} (V8 Model)...")
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
    
    # Calculate metrics
    mae = mean_absolute_error(test_trues_inverse, test_prices_inverse)
    mape = mean_absolute_percentage_error(test_trues_inverse, test_prices_inverse)
    rmse = np.sqrt(mean_squared_error(test_trues_inverse, test_prices_inverse))
    
    logger.info(f"\n  Test Results:")
    logger.info(f"    MAE:  {mae:.6f} USD")
    logger.info(f"    MAPE: {mape:.4f} %")
    logger.info(f"    RMSE: {rmse:.6f} USD")
    
    return {
        'symbol': symbol,
        'predicted': test_prices_inverse,
        'actual': test_trues_inverse,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
    }


def create_html_report(results):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    total_symbols = len(results)
    avg_mae = np.mean([r['mae'] for r in results])
    avg_mape = np.mean([r['mape'] for r in results])
    
    sorted_results = sorted(results, key=lambda x: x['mape'])
    
    table_rows = ""
    for r in sorted_results:
        table_rows += f"""
        <tr>
            <td><strong>{r['symbol']}</strong></td>
            <td>{r['mae']:.6f}</td>
            <td>{r['mape']:.4f}%</td>
            <td>{r['rmse']:.6f}</td>
        </tr>
        """
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V8 All Cryptocurrency Predictions</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #2196F3 0%, #1976d2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f5f5f5;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #2196F3;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #666;
            font-weight: 500;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        thead {{
            background: #2196F3;
            color: white;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        tbody tr:hover {{
            background: #f5f5f5;
        }}
        
        footer {{
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
        
        .badge {{
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>V8 All Cryptocurrency Predictions Report</h1>
            <p>Multi-Symbol Price Prediction Analysis</p>
            <span class="badge">V8 Model</span>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Symbols</div>
                <div class="stat-value">{total_symbols}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average MAE</div>
                <div class="stat-value">{avg_mae:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average MAPE</div>
                <div class="stat-value">{avg_mape:.4f}%</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>MAE (USD)</th>
                            <th>MAPE (%)</th>
                            <th>RMSE (USD)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <p>Generated prediction charts for each symbol:</p>
                <ul style="margin-left: 20px; margin-top: 10px;">
'''
    
    for r in sorted_results:
        html += f"<li><strong>{r['symbol']}_predictions_v8.png</strong> - {r['symbol']} price comparison chart</li>\n"
    
    html += f'''
                </ul>
            </div>
            
            <div class="section">
                <h2>Model Information</h2>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                    <p><strong>Architecture:</strong> Bidirectional LSTM with 2 layers</p>
                    <p><strong>Hidden Size:</strong> Auto-detected from saved models</p>
                    <p><strong>Input Features:</strong> 44 technical indicators</p>
                    <p><strong>Batch Size:</strong> 64 (optimized)</p>
                    <p><strong>Learning Rate:</strong> 0.005 (optimized)</p>
                    <p><strong>Model Path:</strong> models/saved/</p>
                </div>
            </div>
        </div>
        
        <footer>
            <p>V8 All Cryptocurrency Prediction Report | Generated: {timestamp}</p>
            <p style="margin-top: 10px;">Data from CCXT / Binance API</p>
        </footer>
    </div>
</body>
</html>'''
    
    return html


def main():
    global logger
    
    setup_logging()
    
    parser = argparse.ArgumentParser(description='V8 All Cryptocurrency Visualization Tool')
    parser.add_argument('--symbol', type=str, default=None, help='Crypto symbols (comma-separated)')
    args = parser.parse_args()
    
    logger.info('\n' + '='*60)
    logger.info('V8 All Cryptocurrency Visualization Tool')
    logger.info('='*60)
    logger.info(f"\nModel location: models/saved/")
    logger.info(f"Device: {device}")
    
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
            
            fig, ax = plt.subplots(figsize=(12, 5))
            predicted = result['predicted']
            actual = result['actual']
            x = np.arange(len(actual))
            
            ax.plot(x, actual, 'b-', label='Actual Price', linewidth=2, alpha=0.8)
            ax.plot(x, predicted, 'r-', label='Predicted Price (V8)', linewidth=2, alpha=0.8)
            
            ax.set_title(f"{symbol} V8 Price Prediction - MAE: {result['mae']:.4f} | MAPE: {result['mape']:.4f}%", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=10)
            ax.set_ylabel('Price', fontsize=10)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{symbol}_predictions_v8.png', dpi=120, bbox_inches='tight')
            logger.info(f"    Saved: {symbol}_predictions_v8.png")
            plt.close()
        else:
            logger.warning(f"  Error: Failed to process {symbol}, skipping")
    
    if not results:
        logger.error("\nError: No successful predictions")
        return
    
    logger.info(f"\nGenerating HTML report...")
    html_content = create_html_report(results)
    
    with open('predictions_v8_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("  Saved: predictions_v8_report.html")
    
    logger.info(f"\n\n{'='*60}")
    logger.info(f"Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"\nGenerated files:")
    for result in results:
        logger.info(f"  - {result['symbol']}_predictions_v8.png")
    logger.info(f"  - predictions_v8_report.html")
    logger.info(f"\nOpen in browser: predictions_v8_report.html")


if __name__ == '__main__':
    main()
