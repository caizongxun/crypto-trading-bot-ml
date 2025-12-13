#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
數據獲取模块
整合 Binance API 訓練數據獲取
"""

import pandas as pd
import numpy as np
import ccxt
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """從 Binance 獲取加密貨幣數據"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 5000) -> pd.DataFrame:
        """獲取 OHLCV 數據"""
        try:
            trading_pair = f"{symbol}/USDT"
            ohlcv = self.exchange.fetch_ohlcv(trading_pair, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}/{timeframe}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加简專的技術指標 (~40個)"""
        df = df.copy()
        
        # 價格動量
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 移動平均線
        for w in [5, 10, 20, 50]:
            df[f'sma_{w}'] = df['close'].rolling(w).mean()
            df[f'ema_{w}'] = df['close'].ewm(span=w).mean()
        
        # 樣數偏不平方 (Std Dev)
        for w in [5, 10, 20]:
            df[f'std_{w}'] = df['close'].rolling(w).std()
        
        # RSI
        for w in [7, 14]:
            df[f'rsi_{w}'] = self._calculate_rsi(df['close'], w)
        
        # MACD
        macd_vals = self._calculate_macd(df['close'])
        df['macd'] = macd_vals['macd']
        df['macd_signal'] = macd_vals['signal']
        
        # Bollinger Bands
        bb = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb['upper']
        df['bb_lower'] = bb['lower']
        
        # 交易量
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        
        # 價格變化
        for w in [5, 10]:
            df[f'pc_{w}'] = (df['close'] - df['close'].shift(w)) / (df['close'].shift(w) + 1e-8)
        
        # 高低比例
        df['hl_ratio'] = df['high'] / (df['low'] + 1e-8)
        df['ol_ratio'] = df['open'] / (df['low'] + 1e-8)
        
        # 去除 NaN
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Added {len(df.columns) - 6} indicators")
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        deltas = prices.diff()
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / (down + 1e-8)
        rsi = 100. - 100. / (1. + rs)
        rsi[:window] = 0
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return {'macd': macd, 'signal': signal_line}
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> Dict:
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return {
            'upper': sma + (std * num_std),
            'lower': sma - (std * num_std),
            'mid': sma
        }


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_ohlcv('SOL', '1h', 1000)
    df = fetcher.add_indicators(df)
    print(f"Shape: {df.shape}")
    print(df.head())
