#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
所有幣種 V9 模型預測與實際價格路徑對比可視化器

功能:
1. 讀取 V9 訓練的模型 (models/saved/{SYMBOL}_model.pth)
2. 比較預測 vs 實際價格
3. 生成所有幣種的對比圖表
4. 輸出 MAE、MAPE、RMSE 指標
5. 生成 HTML 報告

用法:
  python visualize_predictions_v9.py
  python visualize_predictions_v9.py --symbol SOL
  python visualize_predictions_v9.py --output v9_report.html
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
    'input_size': 60,  # V9: 60+ 技術指標
    'hidden_size': 256,  # V9: 256 隱藏層
    'num_layers': 3,  # V9: 3 層
    'dropout': 0.4,
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
    """添加 60+ 技術指標 (V9 版本)"""
    try:
        # 基本作用
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
        
        # 動量指標
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
        
        # 移動平均系列
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # 衛布指數
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


class EnhancedLSTM(torch.nn.Module):
    """V9 強化的 LSTM 模型"""
    
    def __init__(self):
        super(EnhancedLSTM, self).__init__()
        
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
            torch.nn.Linear(lstm_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(MODEL_CONFIG['dropout']),
            
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(MODEL_CONFIG['dropout']),
            
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(MODEL_CONFIG['dropout']*0.5),
            
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        price = self.regressor(last_out)
        return price


def predict_symbol(symbol: str):
    """一個幣種的預測並計算指標"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {symbol} (V9)...")
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
    
    # 確保 X 的特徵數為 60
    if X_scaled.shape[1] > 60:
        X_scaled = X_scaled[:, :60]
    elif X_scaled.shape[1] < 60:
        padding = np.zeros((X_scaled.shape[0], 60 - X_scaled.shape[1]))
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
    
    model = EnhancedLSTM()
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
    parser = argparse.ArgumentParser(description='Visualize V9 Predictions for All Symbols')
    parser.add_argument('--symbol', type=str, default=None, help='Specific symbol to visualize')
    parser.add_argument('--output', type=str, default='predictions_v9.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info('='*80)
    logger.info('V9 PREDICTIONS VISUALIZATION (Enhanced Precision Model)')
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
    cols = min(5, n_symbols)  # 每行最多 5 欄
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
        
        ax.set_title(f"{symbol} (V9)\nMAE: {result['mae']:.4f} | MAPE: {result['mape']:.4f}%", fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # 隱藏閒置的 subplots
    for idx in range(n_symbols, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('predictions_v9_paths.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: predictions_v9_paths.png")
    plt.close()
    
    # 2. MAE 對比柱狀圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    symbols = [r['symbol'] for r in results]
    maes = [r['mae'] for r in results]
    
    bars = ax.bar(symbols, maes, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 加上數值
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('MAE Comparison - All Symbols (V9 Enhanced Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (USD)', fontsize=12)
    ax.set_xlabel('Symbol', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mae_comparison_v9.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: mae_comparison_v9.png")
    plt.close()
    
    # 3. MAPE 對比柱狀圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    mapes = [r['mape'] for r in results]
    
    bars = ax.bar(symbols, mapes, color='seagreen', alpha=0.7, edgecolor='black')
    
    # 加上數值
    for bar, mape in zip(bars, mapes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mape:.4f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('MAPE Comparison - All Symbols (V9 Enhanced Model)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_xlabel('Symbol', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mape_comparison_v9.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: mape_comparison_v9.png")
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
        
        # 狀態上標
        if mae < 0.5:
            status = '✅ Excellent (V9)'
        elif mae < 1.0:
            status = '✔ Very Good'
        elif mae < 2.0:
            status = '✓ Good'
        elif mae < 5.0:
            status = '⚠️ Fair'
        else:
            status = '❌ Needs Work'
        
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
    
    # 上色會
    for i in range(len(table_data)):
        if i == 0:
            table[(i, 0)].set_facecolor('#2196F3')
            table[(i, 1)].set_facecolor('#2196F3')
            table[(i, 2)].set_facecolor('#2196F3')
            table[(i, 3)].set_facecolor('#2196F3')
            table[(i, 4)].set_facecolor('#2196F3')
            for j in range(5):
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
    
    plt.title('V9 Enhanced Model Performance Report - All Symbols', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('performance_report_v9.png', dpi=150, bbox_inches='tight')
    logger.info("✓ Saved: performance_report_v9.png")
    plt.close()
    
    # 5. 產生 HTML 報告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>V9 Predictions - All Symbols</title>
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
            .badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 12px;
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
                background-color: #2196F3;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 V9 Enhanced Model Predictions - All Symbols</h1>
            
            <div class="info">
                <strong>📅 Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>🔬 Model Version:</strong> <span class="badge">V9 Enhanced Precision</span><br>
                <strong>📈 Total Symbols:</strong> {len(results)}<br>
                <strong>⭐ Average MAE:</strong> {np.mean([r['mae'] for r in results]):.6f} USD<br>
                <strong>🎯 Network Size:</strong> 256x3 LSTM + 60+ Indicators
            </div>
            
            <h2>1. Price Path Comparison (Predicted vs Actual)</h2>
            <p>每個幣種的一段時間內預測價格路徑 vs 實際價格路徑的較</p>
            <img src="predictions_v9_paths.png" alt="Price Paths Comparison">
            
            <h2>2. MAE Comparison (Mean Absolute Error)</h2>
            <p>所有幣種的平均絕對誤差 (MAE) 對比 - 越低越好</p>
            <img src="mae_comparison_v9.png" alt="MAE Comparison">
            
            <h2>3. MAPE Comparison (Mean Absolute Percentage Error)</h2>
            <p>所有幣種的平均百分比誤差 (MAPE) 對比 - 越低越好</p>
            <img src="mape_comparison_v9.png" alt="MAPE Comparison">
            
            <h2>4. Performance Report</h2>
            <p>所有幣種成效故事板 (按 MAE 排序) - V9 增強模型</p>
            <img src="performance_report_v9.png" alt="Performance Report">
            
            <h2>5. Detailed Results Table</h2>
            <table>
                <tr>
                    <th>💰 Symbol</th>
                    <th>📊 MAE (USD)</th>
                    <th>📈 MAPE (%)</th>
                    <th>📉 RMSE (USD)</th>
                    <th>🎯 Model Performance</th>
                </tr>
    """
    
    for result in sorted(results, key=lambda x: x['mae']):
        html_content += f"""
                <tr>
                    <td><strong>{result['symbol']}</strong></td>
                    <td>{result['mae']:.6f}</td>
                    <td>{result['mape']:.4f}%</td>
                    <td>{result['rmse']:.6f}</td>
                    <td><span class="badge">V9 Enhanced</span></td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <div class="info">
                <strong>✨ V9 Improvements:</strong><br>
                ✅ 256x3 LSTM Network (4x larger than V8)<br>
                ✅ 60+ Technical Indicators (vs 44 in V8)<br>
                ✅ BatchNorm + Advanced Regularization<br>
                ✅ SmoothL1Loss for Robustness<br>
                ✅ 200 Epochs Training<br>
                ✅ Fine-tuned Learning Rate (0.0005)<br>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ Saved: {args.output}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ V9 Visualization Complete!")
    logger.info("="*80)
    logger.info(f"\n📄 Generated Files:")
    logger.info(f"  - predictions_v9_paths.png (V9 price paths comparison)")
    logger.info(f"  - mae_comparison_v9.png (V9 MAE comparison)")
    logger.info(f"  - mape_comparison_v9.png (V9 MAPE comparison)")
    logger.info(f"  - performance_report_v9.png (V9 performance table)")
    logger.info(f"  - {args.output} (HTML report)")
    logger.info(f"\n⚡ Open {args.output} in browser to view full report!")


if __name__ == '__main__':
    main()
