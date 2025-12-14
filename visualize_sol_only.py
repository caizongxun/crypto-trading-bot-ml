#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOL 子編程可視化器 - 快速測試单個幣種

用法:
  python visualize_sol_only.py
  python visualize_sol_only.py --epochs 50  # 快速測試
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
    'input_size': 60,
    'hidden_size': 256,
    'num_layers': 3,
    'dropout': 0.4,
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


def fetch_data(symbol: str, timeframe: str = '1h', limit: int = 1000):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"📊 接取 {limit} 根蹫樗價 {symbol}/{timeframe}...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ 接取完成 {len(df)} 根蹫樗價")
        return df
    
    except Exception as e:
        logger.error(f"❌ 接取數據失敗: {e}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['high-low'] = df['high'] - df['low']
        df['close-open'] = df['close'] - df['open']
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = macd.ewm(span=signal).mean()
            df[f'macd_hist_{fast}_{slow}'] = macd - df[f'macd_signal_{fast}_{slow}']
        
        for period in [20, 50]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].diff(period)
        
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        
        for period in [5, 10]:
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
        
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_sma'] = obv.rolling(window=20).mean()
        
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                           (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * (-100)
        
        df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
        
        df['money_flow'] = (df['close'] - df['open']) * df['volume']
        df['money_flow_sma'] = df['money_flow'].rolling(window=20).mean()
        
        df = df.ffill().bfill()
        
        logger.info(f"✓ 添加了 {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} 個技术指標")
        return df
    
    except Exception as e:
        logger.error(f"❌ 添加技术指標失敗: {e}")
        return None


def prepare_sequences(X, y, lookback=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class EnhancedLSTM(torch.nn.Module):
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


def predict_sol():
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 處理 SOL (V9 增強模型)...")
    logger.info(f"{'='*60}")
    
    # 接取數據
    df = fetch_data('SOL')
    if df is None or len(df) == 0:
        logger.error(f"❌ 接取 SOL 數據失敗")
        return None
    
    # 添加技术指標
    df = add_technical_indicators(df)
    if df is None:
        logger.error(f"❌ 添加技术指標失敗")
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
    
    logger.info(f"✓ 特徵矩陣: {X_scaled.shape}")
    
    # 準備序列
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, 60)
    
    # train/val/test 分割
    n_samples = len(X_seq)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    logger.info(f"✓ 數據分割: Train={train_size}, Val={val_size}, Test={len(X_test)}")
    
    # 加載模型
    model_path = f'models/saved/SOL_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"❌ 找不到 SOL 模型: {model_path}")
        logger.info(f"\n💡 請先訓練 SOL V9 模型:")
        logger.info(f"   python training/train_lstm_v9_precision.py --symbol SOL --epochs 200")
        return None
    
    model = EnhancedLSTM()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info(f"✓ 模型加載成功")
    
    # 預測
    with torch.no_grad():
        test_prices = []
        test_trues = []
        
        for i in range(0, len(X_test), 32):
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
    
    logger.info(f"\n📊 性能指標:")
    logger.info(f"  MAE:  {mae:.6f} USD")
    logger.info(f"  MAPE: {mape:.4f} %")
    logger.info(f"  RMSE: {rmse:.6f} USD")
    
    return {
        'symbol': 'SOL',
        'predicted': test_prices_inverse,
        'actual': test_trues_inverse,
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
    }


def main():
    global logger
    
    setup_logging()
    
    logger.info('\n' + '='*60)
    logger.info('SOL 個平霹化器 - V9 模型')
    logger.info('='*60)
    
    result = predict_sol()
    
    if not result:
        logger.error("\n❌ 失敗")
        return
    
    # 產生可視化
    logger.info(f"\n📈 生成可視化...")
    
    # 1. 價格路徑對比
    fig, ax = plt.subplots(figsize=(14, 6))
    
    predicted = result['predicted']
    actual = result['actual']
    x = np.arange(len(actual))
    
    ax.plot(x, actual, 'b-', label='Actual Price', linewidth=2.5, alpha=0.8)
    ax.plot(x, predicted, 'r-', label='Predicted Price (V9)', linewidth=2.5, alpha=0.8)
    
    ax.set_title(f"SOL V9 价格預測 - MAE: {result['mae']:.4f} USD | MAPE: {result['mape']:.4f}%", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('SOL_predictions_v9.png', dpi=150, bbox_inches='tight')
    logger.info("✓ 保存: SOL_predictions_v9.png")
    plt.close()
    
    # 2. 統計粗
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['MAE (USD)', 'MAPE (%)', 'RMSE (USD)']
    values = [result['mae'], result['mape'], result['rmse']]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_title('SOL V9 性能指標', fontsize=14, fontweight='bold')
    ax.set_ylabel('值', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('SOL_metrics_v9.png', dpi=150, bbox_inches='tight')
    logger.info("✓ 保存: SOL_metrics_v9.png")
    plt.close()
    
    # 3. 生成 HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SOL V9 預測</title>
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
                max-width: 1000px;
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
            
            header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #2196F3;
            }}
            
            .stat-value {{
                font-size: 1.8em;
                font-weight: bold;
                color: #2196F3;
                margin: 10px 0;
            }}
            
            .stat-label {{
                color: #666;
                font-weight: 500;
            }}
            
            .section {{
                margin-bottom: 30px;
            }}
            
            .section h2 {{
                color: #333;
                margin-bottom: 15px;
                font-size: 1.5em;
                border-bottom: 2px solid #2196F3;
                padding-bottom: 10px;
            }}
            
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                margin: 15px 0;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}
            
            footer {{
                background: #f5f5f5;
                padding: 20px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
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
                <h1>📊 SOL V9 模型預測</h1>
                <p>Solana Price Prediction - V9 Enhanced Model</p>
                <span class="badge">V9 增強模型</span>
            </header>
            
            <div class="content">
                <div class="section">
                    <h2>📈 性能指標</h2>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-label">平均綕對誤差</div>
                            <div class="stat-value">{result['mae']:.6f}</div>
                            <div class="stat-label">USD</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">平均百分比誤差</div>
                            <div class="stat-value">{result['mape']:.4f}</div>
                            <div class="stat-label">%</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">根平方誤差</div>
                            <div class="stat-value">{result['rmse']:.6f}</div>
                            <div class="stat-label">USD</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📉 價格路徑對比</h2>
                    <p>藍線 = 實際價格 | 紅線 = V9 預測價格</p>
                    <img src="SOL_predictions_v9.png" alt="SOL Price Predictions">
                </div>
                
                <div class="section">
                    <h2>📄 指標比較</h2>
                    <img src="SOL_metrics_v9.png" alt="SOL Metrics">
                </div>
                
                <div class="section">
                    <h2>ℹ️ 模型資誊</h2>
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;">
                        <p><strong>🎯 网络結構:</strong> 256 隱藏 x 3 層</p>
                        <p><strong>📋 技术指標:</strong> 60+ 個</p>
                        <p><strong>📄 訓練 Epochs:</strong> 200</p>
                        <p><strong>📉 Loss 函數:</strong> SmoothL1Loss</p>
                        <p><strong>✨ 技訓:</strong> BatchNorm + AdamW 優化</p>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>📊 SOL V9 預測 | 生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p style="margin-top: 10px;">Data from CCXT / Binance API</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    with open('SOL_predictions_v9.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("✓ 保存: SOL_predictions_v9.html")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 完成！")
    logger.info("="*60)
    logger.info(f"\n📄 生成的文件:")
    logger.info(f"  - SOL_predictions_v9.png (价格路径对比)")
    logger.info(f"  - SOL_metrics_v9.png (性能指标)")
    logger.info(f"  - SOL_predictions_v9.html (HTML报告)")
    logger.info(f"\n🌐 在浏览器中打开: SOL_predictions_v9.html")


if __name__ == '__main__':
    main()
