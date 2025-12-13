#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理引擎 - 載入本地訓練的模型並預測價格
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import ccxt  # Binance API

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """与 train_lstm_v1.py 一樣的 LSTM 模型"""
    
    def __init__(self, input_size=44, hidden_size=128, num_layers=2, output_size=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2  # bidirectional
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


class CryptoPredictor:
    """加密貨幣價格預測器"""
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}  # 缓存已載入的模型
        self.exchange = ccxt.binance()  # Binance API
        
        logger.info(f"Predictor initialized on {self.device}")
        logger.info(f"Model directory: {self.model_dir}")
    
    def load_model(self, symbol: str) -> Optional[LSTMModel]:
        """載入模型"""
        if symbol in self.models:
            return self.models[symbol]
        
        model_path = self.model_dir / f"{symbol}_model.pth"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            model = LSTMModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models[symbol] = model
            logger.info(f"Loaded model for {symbol}")
            
            return model
        
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            return None
    
    def fetch_recent_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """從 Binance 獲取最近 K 线數據"""
        try:
            trading_pair = f"{symbol}/USDT"
            ohlcv = self.exchange.fetch_ohlcv(trading_pair, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取技術指標（樣例實裝）"""
        # TODO: 實裝 44 個技術指標
        # 此處简化為目前只提取 OHLCV + 低度指標
        
        features = []
        
        # Basic OHLCV
        features.append(df[['open', 'high', 'low', 'close', 'volume']].values)
        
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        features.append(df['sma_20'].values.reshape(-1, 1))
        
        # ... 更多指標可以加入
        
        return np.concatenate(features, axis=1)
    
    def predict(self, symbol: str) -> Dict:
        """預測一個幣種的下一小時價格"""
        model = self.load_model(symbol)
        if model is None:
            return {'error': f'Model not found for {symbol}'}
        
        try:
            # 1. 獲取數據
            df = self.fetch_recent_data(symbol)
            if df is None or len(df) < 60:
                return {'error': 'Insufficient data'}
            
            # 2. 提取特徵
            features = self.extract_features(df)
            
            # 3. 正見化（可選）
            # 岍短程LSTM模型通常已樣業推理提供平正見化推理
            
            # 4. 操作模型
            X = torch.tensor(features[-60:], dtype=torch.float32).unsqueeze(0)
            X = X.to(self.device)
            
            with torch.no_grad():
                pred = model(X)
                pred_price = pred.item()
            
            current_price = float(df['close'].iloc[-1])
            change_pct = (pred_price - current_price) / current_price * 100
            
            # 粗続信心度
            confidence = min(95, max(50, 90 + (50 - abs(change_pct))))
            
            # 信號
            if change_pct > 0.5:
                signal = "📈 BUY"
            elif change_pct < -0.5:
                signal = "📉 SELL"
            else:
                signal = "🔄 HOLD"
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': pred_price,
                'change_percent': change_pct,
                'confidence': confidence,
                'signal': signal,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {str(e)}")
            return {'error': str(e)}


if __name__ == '__main__':
    predictor = CryptoPredictor()
    result = predictor.predict('SOL')
    print(result)
