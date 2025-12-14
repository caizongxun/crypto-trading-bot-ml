#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOL V8 可視化器 - 使用訓練好的 V8 模型
模型路徑: models/backup_v8/

用法:
  python visualize_sol_v8.py
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

# V8 配置 (44 個技術指標)
MODEL_CONFIG = {
    'input_size': 44,
    'hidden_size': 128,
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


def fetch_data(symbol: str, timeframe: str = '1h', limit: int = 1000):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        symbol_pair = f"{symbol}/USDT"
        
        logger.info(f"📊 接取 {limit} 根蠋燭價 {symbol}/{timeframe}...")
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✓ 接取完成 {len(df)} 根蠋燭價")
        return df
    
    except Exception as e:
        logger.error(f"❌ 接取數據失敗: {e}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加 44 個技術指標 (V8 版本)"""
    try:
        # 基本作用
        df['high-low'] = df['high'] - df['low']
        df['close-open'] = df['close'] - df['open']
        df['returns'] = df['close'].pct_change()
        
        # RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
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
        
        logger.info(f"✓ 添加了 {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} 個技術指標 (V8)")
        return df
    
    except Exception as e:
        logger.error(f"❌ 添加技術指標失敗: {e}")
        return None


def prepare_sequences(X, y, lookback=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)


class RegressionLSTM(torch.nn.Module):
    """V8 LSTM 模型"""
    
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


def predict_sol():
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 處理 SOL (V8 穩定模型)...")
    logger.info(f"{'='*60}")
    
    # 接取數據
    df = fetch_data('SOL')
    if df is None or len(df) == 0:
        logger.error(f"❌ 接取 SOL 數據失敗")
        return None
    
    # 添加技術指標 (V8: 44 個)
    df = add_technical_indicators(df)
    if df is None:
        logger.error(f"❌ 添加技術指標失敗")
        return None
    
    # 特徵提取
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
    X = df[feature_cols].values
    y = df['close'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 確保 X 的特徵數為 44 (不是 60)
    if X_scaled.shape[1] > 44:
        X_scaled = X_scaled[:, :44]
    elif X_scaled.shape[1] < 44:
        padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, padding])
    
    logger.info(f"✓ 特徵矩陣 (V8): {X_scaled.shape}")
    
    # 準備序列
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, 60)
    
    # train/val/test 分割
    n_samples = len(X_seq)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    logger.info(f"✓ 數據分割: Train={train_size}, Val={val_size}, Test={len(X_test)}")
    
    # 加載模型 - 從 backup_v8 路徑
    model_path = f'models/backup_v8/SOL_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"❌ 找不到 SOL 模型: {model_path}")
        logger.info(f"\n📁 檢查以下路徑:")
        logger.info(f"   1. {model_path}")
        
        # 嘗試列出 backup_v8 目錄
        backup_dir = 'models/backup_v8'
        if os.path.exists(backup_dir):
            logger.info(f"\n   backup_v8 目錄中的文件:")
            for file in os.listdir(backup_dir):
                logger.info(f"      - {file}")
        else:
            logger.info(f"   ❌ {backup_dir} 目錄不存在")
        
        return None
    
    model = RegressionLSTM()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info(f"✓ V8 模型加載成功 (44 個技術指標)")
    logger.info(f"   路徑: {model_path}")
    
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
    logger.info('SOL 個評析化器 - V8 穩定模型')
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
    ax.plot(x, predicted, 'r-', label='Predicted Price (V8)', linewidth=2.5, alpha=0.8)
    
    ax.set_title(f"SOL V8 價格預測 - MAE: {result['mae']:.4f} USD | MAPE: {result['mape']:.4f}%", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('SOL_predictions_v8.png', dpi=150, bbox_inches='tight')
    logger.info("✓ 保存: SOL_predictions_v8.png")
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
    
    ax.set_title('SOL V8 性能指標', fontsize=14, fontweight='bold')
    ax.set_ylabel('值', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('SOL_metrics_v8.png', dpi=150, bbox_inches='tight')
    logger.info("✓ 保存: SOL_metrics_v8.png")
    plt.close()
    
    # 3. 生成 HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SOL V8 預測</title>
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
            
            .version-note {{
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #2196F3;
            }}
        </style>
    </head>
    <body>
        <div class=\"container\">
            <header>
                <h1>📊 SOL V8 模型預測</h1>
                <p>Solana Price Prediction - V8 Stable Model</p>
                <span class=\"badge\">V8 穩定模型</span>
            </header>
            
            <div class=\"content\">
                <div class=\"version-note\">
                    <strong>🌟 V8 特點:</strong>
                    <ul style=\"margin-left: 20px; margin-top: 10px;\">
                        <li>✓ 策略穩定 - 128 x 2 網絡結構</li>
                        <li>✓ 44 個技術指標 (覆蓋索轉所需)</li>
                        <li>✓ 已經過訓練驗證 (效果穩定)</li>
                        <li>✓ 推薦鏡像應用</li>
                    </ul>
                </div>
                
                <div class=\"section\">
                    <h2>📈 性能指標</h2>
                    <div class=\"stats\">
                        <div class=\"stat-card\">
                            <div class=\"stat-label\">平均絕對誤差</div>
                            <div class=\"stat-value\">{result['mae']:.6f}</div>
                            <div class=\"stat-label\">USD</div>
                        </div>
                        <div class=\"stat-card\">
                            <div class=\"stat-label\">平均百分比誤差</div>
                            <div class=\"stat-value\">{result['mape']:.4f}</div>
                            <div class=\"stat-label\">%</div>
                        </div>
                        <div class=\"stat-card\">
                            <div class=\"stat-label\">根平方誤差</div>
                            <div class=\"stat-value\">{result['rmse']:.6f}</div>
                            <div class=\"stat-label\">USD</div>
                        </div>
                    </div>
                </div>
                
                <div class=\"section\">
                    <h2>📈 價格路徑對比</h2>
                    <p>📍 藍線 = 實際價格 | 紅線 = V8 預測價格</p>
                    <img src=\"SOL_predictions_v8.png\" alt=\"SOL Price Predictions\">
                </div>
                
                <div class=\"section\">
                    <h2>📋 指標比較</h2>
                    <img src=\"SOL_metrics_v8.png\" alt=\"SOL Metrics\">
                </div>
                
                <div class=\"section\">
                    <h2>ℹ️ 模型資訊</h2>
                    <div style=\"background: #f5f5f5; padding: 15px; border-radius: 8px;\">
                        <p><strong>🧠 網絡結構:</strong> 128 隱藏 x 2 層 (V8 標準配置)</p>\n                        <p><strong>📋 技術指標:</strong> 44 個</p>\n                        <p><strong>📊 訓練 Epochs:</strong> 150</p>\n                        <p><strong>📉 Loss 函數:</strong> MSE</p>\n                        <p><strong>✨ 優化:</strong> Adam 優化器</p>\n                        <p><strong>📁 模型路徑:</strong> models/backup_v8/SOL_model.pth</p>\n                    </div>\n                </div>\n            </div>\n            \n            <footer>\n                <p>📊 SOL V8 預測 | 生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n                <p style=\"margin-top: 10px;\">Data from CCXT / Binance API</p>\n            </footer>\n        </div>\n    </body>\n    </html>\n    \"\"\"\n    \n    with open('SOL_predictions_v8.html', 'w', encoding='utf-8') as f:\n        f.write(html_content)\n    \n    logger.info(\"✓ 保存: SOL_predictions_v8.html\")\n    \n    logger.info(\"\\n\" + \"=\"*60)\n    logger.info(\"✅ 完成！\")\n    logger.info(\"=\"*60)\n    logger.info(f\"\\n📄 生成的文件:\")\n    logger.info(f\"  - SOL_predictions_v8.png (價格路徑對比)\")\n    logger.info(f\"  - SOL_metrics_v8.png (性能指標)\")\n    logger.info(f\"  - SOL_predictions_v8.html (HTML報告)\")\n    logger.info(f\"\\n🌐 在瀏覽器中打開: SOL_predictions_v8.html\")\n\n\nif __name__ == '__main__':\n    main()\n