#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bot Predictor V8 - For Discord Bot Integration

Direct prediction module for trading signals with automatic bias correction

Usage:
  from bot_predictor import BotPredictor
  
  bot = BotPredictor()
  prediction = bot.predict('BTC')
  print(f"Corrected Price: {prediction['corrected_price']}")
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import ccxt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RegressionLSTM(torch.nn.Module):
    """V8 LSTM Model"""
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


class BotPredictor:
    """Bot Prediction Engine with Bias Correction"""
    
    def __init__(self, model_dir='models/saved', bias_config_path='models/bias_corrections_v8.json'):
        self.model_dir = model_dir
        self.device = device
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.model_cache = {}
        self.scaler_cache = {}
        
        # Load bias corrections
        self.bias_corrections = {}
        if os.path.exists(bias_config_path):
            try:
                with open(bias_config_path, 'r') as f:
                    bias_config = json.load(f)
                    self.bias_corrections = bias_config.get('corrections', {})
                    logger.info(f"Loaded bias corrections for {len(self.bias_corrections)} symbols")
            except Exception as e:
                logger.warning(f"Could not load bias corrections: {e}")
    
    def _detect_model_config(self, state_dict):
        """Detect model architecture from weights"""
        try:
            weight_ih = state_dict.get('lstm.weight_ih_l0')
            hidden_size = weight_ih.shape[0] // 4 if weight_ih is not None else 64
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
    
    def _fetch_data(self, symbol, limit=1000):
        """Fetch latest OHLCV data"""
        try:
            symbol_pair = f"{symbol}/USDT"
            ohlcv = self.exchange.fetch_ohlcv(symbol_pair, '1h', limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def _add_indicators(self, df):
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
            logger.error(f"Error adding indicators: {e}")
            return None
    
    def _load_model(self, symbol):
        """Load model from cache or disk"""
        if symbol in self.model_cache:
            return self.model_cache[symbol]
        
        # Find model file
        possible_names = [f'{symbol}_model_v8.pth', f'{symbol}_model.pth', f'{symbol}.pth']
        model_path = None
        
        for name in possible_names:
            path = os.path.join(self.model_dir, name)
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            logger.error(f"Model not found for {symbol}")
            return None
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            config = self._detect_model_config(state_dict)
            
            model = RegressionLSTM(
                input_size=44,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional']
            )
            model.to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            self.model_cache[symbol] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return None
    
    def predict(self, symbol, apply_correction=True):
        """
        Predict next price for symbol
        
        Returns:
            dict with keys:
            - raw_price: 未校正的預測價格
            - correction: 校正值
            - corrected_price: 校正後的預測價格 (推薦用這個)
            - current_price: 當前價格
            - direction: 'UP' 或 'DOWN'
            - confidence: 0-1 信心指數
        """
        try:
            # Fetch data
            df = self._fetch_data(symbol)
            if df is None or len(df) < 100:
                logger.error(f"Insufficient data for {symbol}")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Add indicators
            df = self._add_indicators(df)
            if df is None:
                return None
            
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
            X = df[feature_cols].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            scaler_X = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            
            if X_scaled.shape[1] > 44:
                X_scaled = X_scaled[:, :44]
            elif X_scaled.shape[1] < 44:
                padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
            
            # Prepare sequence
            lookback = 60
            if len(X_scaled) < lookback + 1:
                logger.error(f"Insufficient sequence data for {symbol}")
                return None
            
            X_seq = X_scaled[-lookback:].reshape(1, lookback, 44)
            
            # Load model and predict
            model = self._load_model(symbol)
            if model is None:
                return None
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
                price_scaled = model(X_tensor).cpu().numpy()[0][0]
            
            # Inverse transform price
            y_scaler = MinMaxScaler()
            y_scaler.fit(df['close'].values.reshape(-1, 1))
            raw_price = y_scaler.inverse_transform([[price_scaled]])[0][0]
            
            # Apply bias correction
            correction = self.bias_corrections.get(symbol, 0)
            corrected_price = raw_price + correction if apply_correction else raw_price
            
            # Direction
            direction = 'UP' if corrected_price > current_price else 'DOWN'
            change_pct = abs(corrected_price - current_price) / current_price * 100
            confidence = min(change_pct / 2, 1.0)  # Simple confidence metric
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'raw_price': float(raw_price),
                'correction': float(correction),
                'corrected_price': float(corrected_price),
                'direction': direction,
                'change_pct': float(change_pct),
                'confidence': float(confidence),
                'model_version': 'v8',
            }
        
        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None


if __name__ == '__main__':
    # Test
    bot = BotPredictor()
    
    test_symbols = ['BTC', 'ETH', 'SOL']
    for symbol in test_symbols:
        prediction = bot.predict(symbol)
        if prediction:
            print(f"\n{symbol}:")
            print(f"  Current: ${prediction['current_price']:.2f}")
            print(f"  Predicted: ${prediction['corrected_price']:.2f}")
            print(f"  Direction: {prediction['direction']}")
            print(f"  Confidence: {prediction['confidence']*100:.1f}%")
