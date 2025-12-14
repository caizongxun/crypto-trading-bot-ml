#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
所有幣種預測与實際價格路徑對比可視化器

功能:
1. 每個幣種標極了上迷的預測与實際價格
2. 所有幣種的 MAE 對比
3. 所有幣種的 MAPE 對比
4. 所有幣種的故事板
用法:
  python visualize_predictions_v8.py
  python visualize_predictions_v8.py --symbol SOL
  python visualize_predictions_v8.py --output report.html
"""

import os
import sys
import io
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 非 GUI 統訪

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import ccxt
import logging

# 設定 Windows UTF-8 編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 所有幣種
SYMBOLS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "DOT", "AVAX", "LINK",
           "UNI", "LTC", "MATIC", "ARB", "OP", "ATOM", "FTM", "NEAR", "PEPE", "SHIB"]

MODEL_CONFIG = {
    'input_size': 44,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'bidirectional': True,
}


def setup_logging():
    """設定日誌"""
    global logger
    
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


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
        
        df = df.ffill()
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return None


def prepare_sequences(X, y, lookback=60):
    """整理序列"""
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class RegressionLSTM(torch.nn.Module):
    """V8 回歸 LSTM 模型"""
    
    def __init__(self):
        super(RegressionLSTM, self).__init__()
        
        self.lstm = torch.nn.LSTM(
            input_size=MODEL_CONFIG['input_size'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout'],
            bidirectional=MODEL_CONFIG['bidirectional'],
            batch_first=True
        )
        
        lstm_output_size = MODEL_CONFIG['hidden_size'] * 2
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(lstm_output_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(MODEL_CONFIG['dropout']),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        price = self.regressor(last_out)
        return price


def predict_symbol(symbol: str):
    """一個幣種的預測並計算指標"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {symbol}...")
    logger.info(f"{'='*60}")
    
    # 獲取數據
    df = fetch_data(symbol)
    if df is None or len(df) == 0:
        logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    # 添加技術指標
    df = add_technical_indicators(df)
    if df is None:
        logger.error(f"Failed to add indicators for {symbol}")
        return None
    
    # 特徵提取
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
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
    
    # 準備序列
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, 60)
    
    # train/val/test 分割
    n_samples = len(X_seq)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    # 加載模型
    model_path = f'models/saved/{symbol}_model.pth'
    if not os.path.exists(model_path):
        logger.warning(f"Model not found for {symbol}: {model_path}")
        return None
    
    model = RegressionLSTM()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 預測
    with torch.no_grad():
        test_prices = []
        test_trues = []
        
        for i in range(0, len(X_test), 32):  # batch size 32
            X_batch = torch.tensor(X_test[i:i+32]).to(device).float()
            price = model(X_batch)
            test_prices.extend(price.cpu().numpy().flatten())
            test_trues.extend(y_test[i:i+32])
    
    # 反正規化
    test_prices_inverse = scaler_y.inverse_transform(np.array(test_prices).reshape(-1, 1)).flatten()
    test_trues_inverse = scaler_y.inverse_transform(np.array(test_trues).reshape(-1, 1)).flatten()
    
    # 計算指標
    mae = mean_absolute_error(test_trues_inverse, test_prices_inverse)
    mape = mean_absolute_percentage_error(test_trues_inverse, test_prices_inverse)
    rmse = np.sqrt(mean_squared_error(test_trues_inverse, test_prices_inverse))
    
    logger.info(f"MAE:  {mae:.6f} USD")
    logger.info(f"MAPE: {mape:.4f} %")
    logger.info(f"RMSE: {rmse:.6f} USD")
    
    return {
        'symbol': symbol,
        'predicted': test_prices_inverse,
        'actual': test_trues_inverse,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
    }


def main():
    global logger
    
    import argparse
    parser = argparse.ArgumentParser(description='Visualize V8 Predictions for All Symbols')
    parser.add_argument('--symbol', type=str, default=None, help='Specific symbol to visualize')
    parser.add_argument('--output', type=str, default='predictions_v8.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info('='*80)
    logger.info('V8 PREDICTIONS VISUALIZATION')
    logger.info('='*80)
    
    symbols_to_process = [args.symbol] if args.symbol else SYMBOLS
    
    results = []
    for symbol in symbols_to_process:
        result = predict_symbol(symbol)
        if result:
            results.append(result)
    
    if not results:
        logger.error("No results generated!")
        return
    
    # 產生視覺化
    logger.info(f"\nGenerating visualizations...")
    
    # 1. 每個幣種的價格路徑對比
    n_symbols = len(results)
    cols = min(5, n_symbols)  # 每行最多 5 穱
    rows = (n_symbols + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    axes = axes.flatten() if n_symbols > 1 else [axes]
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        symbol = result['symbol']
        predicted = result['predicted']
        actual = result['actual']
        
        x = np.arange(len(actual))
        
        ax.plot(x, actual, 'b-', label='Actual', linewidth=2, alpha=0.7)
        ax.plot(x, predicted, 'r-', label='Predicted', linewidth=2, alpha=0.7)
        
        ax.set_title(f"{symbol}\nMAE: {result['mae']:.4f} | MAPE: {result['mape']:.4f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # 隱藏婗余的 subplots
    for idx in range(n_symbols, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('predictions_v8_paths.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: predictions_v8_paths.png")
    plt.close()
    
    # 2. MAE 對比柱狀圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    symbols = [r['symbol'] for r in results]
    maes = [r['mae'] for r in results]
    
    bars = ax.bar(symbols, maes, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 加上數佐
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('MAE Comparison - All Symbols (V8)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (USD)', fontsize=12)
    ax.set_xlabel('Symbol', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mae_comparison_v8.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: mae_comparison_v8.png")
    plt.close()
    
    # 3. MAPE 對比柱狀圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    mapes = [r['mape'] for r in results]
    
    bars = ax.bar(symbols, mapes, color='seagreen', alpha=0.7, edgecolor='black')
    
    # 加上數佐
    for bar, mape in zip(bars, mapes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mape:.4f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('MAPE Comparison - All Symbols (V8)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_xlabel('Symbol', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mape_comparison_v8.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: mape_comparison_v8.png")
    plt.close()
    
    # 4. 故事板
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = [
        ['Symbol', 'MAE (USD)', 'MAPE (%)', 'RMSE (USD)', 'Status']
    ]
    
    for result in sorted(results, key=lambda x: x['mae']):
        symbol = result['symbol']
        mae = result['mae']
        mape = result['mape']
        rmse = result['rmse']
        
        # 狀態上迷
        if mae < 1.0:
            status = '✅ Excellent'
        elif mae < 2.0:
            status = '✌ Good'
        elif mae < 5.0:
            status = '⚠️ Fair'
        else:
            status = '❌ Poor'
        
        table_data.append([
            symbol,
            f"{mae:.6f}",
            f"{mape:.4f}%",
            f"{rmse:.6f}",
            status
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 上色会
    for i in range(len(table_data)):
        if i == 0:
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 2)].set_facecolor('#4CAF50')
            table[(i, 3)].set_facecolor('#4CAF50')
            table[(i, 4)].set_facecolor('#4CAF50')
            for j in range(5):
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('V8 Model Performance Report - All Symbols', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('performance_report_v8.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: performance_report_v8.png")
    plt.close()
    
    # 5. 產生 HTML 報告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>V8 Predictions - All Symbols</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border-radius: 8px;
            }}
            .info {{
                background-color: #e3f2fd;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                border-left: 4px solid #2196F3;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 V8 Predictions Visualization - All Symbols</h1>
            
            <div class="info">
                <strong>📅 Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>📊 Total Symbols:</strong> {len(results)}<br>
                <strong>🌟 Average MAE:</strong> {np.mean([r['mae'] for r in results]):.6f} USD
            </div>
            
            <h2>1. Price Path Comparison (Predicted vs Actual)</h2>
            <p>每個幣種的一段時間內預測價格路徑 vs 實際價格路徑的較较</p>
            <img src="predictions_v8_paths.png" alt="Price Paths Comparison">
            
            <h2>2. MAE Comparison</h2>
            <p>所有幣種的平均綕對誤差 (MAE) 對比</p>
            <img src="mae_comparison_v8.png" alt="MAE Comparison">
            
            <h2>3. MAPE Comparison</h2>
            <p>所有幣種的平均綕對省數誤差 (MAPE) 對比</p>
            <img src="mape_comparison_v8.png" alt="MAPE Comparison">
            
            <h2>4. Performance Report</h2>
            <p>所有幣種成效故事板 (按 MAE 排序)</p>
            <img src="performance_report_v8.png" alt="Performance Report">
            
            <h2>5. Detailed Results</h2>
            <table>
                <tr>
                    <th>💋 Symbol</th>
                    <th>📋 MAE (USD)</th>
                    <th>📊 MAPE (%)</th>
                    <th>🌟 RMSE (USD)</th>
                </tr>
    """
    
    for result in sorted(results, key=lambda x: x['mae']):
        html_content += f"""
                <tr>
                    <td><strong>{result['symbol']}</strong></td>
                    <td>{result['mae']:.6f}</td>
                    <td>{result['mape']:.4f}%</td>
                    <td>{result['rmse']:.6f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ Saved: {args.output}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Visualization Complete!")
    logger.info("="*80)
    logger.info(f"\n📄 Generated Files:")
    logger.info(f"  - predictions_v8_paths.png (Price paths comparison)")
    logger.info(f"  - mae_comparison_v8.png (MAE comparison)")
    logger.info(f"  - mape_comparison_v8.png (MAPE comparison)")
    logger.info(f"  - performance_report_v8.png (Performance table)")
    logger.info(f"  - {args.output} (HTML report)")
    logger.info(f"\n⚡ Open {args.output} in browser to view full report!")


if __name__ == '__main__':
    main()
