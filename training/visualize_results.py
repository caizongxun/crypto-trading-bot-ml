#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可視化工具 - 展示模型預測準確度與價格線的對比
生成多種圖表幫助評估模型性能
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, accuracy_score

# 設定 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設定圖表風格
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
sns.set_style("darkgrid")


class ModelVisualizer:
    """模型可視化類"""
    
    def __init__(self, symbol: str, model_path: str = None, results_path: str = None):
        self.symbol = symbol
        self.model_path = model_path or f'models/saved/{symbol}_model.pth'
        self.results_path = results_path or f'results/{symbol}_results.json'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_results(self) -> dict:
        """載入訓練結果"""
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ 結果文件不存在: {self.results_path}")
            return None
    
    def fetch_and_predict(self, limit: int = 200):
        """獲取數據並進行預測"""
        try:
            import ccxt
            import yaml
            
            # 動態導入訓練模塊
            from train_lstm_v1 import (
                LSTMModel, add_technical_indicators, 
                prepare_sequences, calculate_direction_accuracy
            )
            
            # 載入配置
            with open('training/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 獲取數據
            exchange = ccxt.binance()
            trading_pair = f"{self.symbol}/USDT"
            ohlcv = exchange.fetch_ohlcv(trading_pair, '1h', limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 添加技術指標
            df = add_technical_indicators(df)
            
            # 特徵提取
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
            X = df[feature_cols].values
            y = df['close'].values
            
            # 正規化
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
            X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, config['training']['lookback_window'])
            
            # 載入模型
            model = LSTMModel(config)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # 預測
            y_preds = []
            with torch.no_grad():
                for x in X_seq:
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                    pred = model(x_tensor)
                    y_preds.append(pred.cpu().numpy()[0, 0])
            
            y_preds = np.array(y_preds)
            
            # 反正規化
            y_true_inverse = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).flatten()
            y_pred_inverse = scaler_y.inverse_transform(y_preds.reshape(-1, 1)).flatten()
            
            return {
                'df': df,
                'y_true': y_true_inverse,
                'y_pred': y_pred_inverse,
                'timestamps': df['timestamp'].iloc[config['training']['lookback_window']:].values,
                'config': config
            }
        
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_visualizations(self, pred_data: dict):
        """建立多個可視化圖表"""
        if pred_data is None:
            print("❌ 沒有預測數據可用")
            return
        
        y_true = pred_data['y_true']
        y_pred = pred_data['y_pred']
        timestamps = pred_data['timestamps']
        
        # 計算指標
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        dir_acc = self._calculate_direction_accuracy(y_true, y_pred)
        
        # 建立圖表
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(f'{self.symbol} LSTM 模型預測準確度分析 - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. 價格預測 vs 實際價格
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(timestamps, y_true, label='實際價格', color='#1f77b4', linewidth=2, alpha=0.8)
        ax1.plot(timestamps, y_pred, label='預測價格', color='#ff7f0e', linewidth=1.5, alpha=0.7, linestyle='--')
        ax1.fill_between(timestamps, y_true, y_pred, alpha=0.2, color='gray')
        ax1.set_xlabel('時間')
        ax1.set_ylabel('價格 (USDT)')
        ax1.set_title(f'價格預測對比 (MAE: ${mae:.4f})', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 誤差分布
        ax2 = plt.subplot(3, 2, 2)
        errors = np.abs(y_true - y_pred)
        ax2.hist(errors, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax2.axvline(mae, color='red', linestyle='--', linewidth=2, label=f'平均誤差: ${mae:.4f}')
        ax2.set_xlabel('絕對誤差 (USD)')
        ax2.set_ylabel('頻率')
        ax2.set_title('誤差分布', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 預測 vs 實際散點圖
        ax3 = plt.subplot(3, 2, 3)
        ax3.scatter(y_true, y_pred, alpha=0.5, s=20, color='#d62728')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='完美預測線')
        ax3.set_xlabel('實際價格 (USD)')
        ax3.set_ylabel('預測價格 (USD)')
        ax3.set_title(f'預測 vs 實際 (R²: {r2:.4f})', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 誤差時間序列
        ax4 = plt.subplot(3, 2, 4)
        errors = y_true - y_pred
        colors = ['green' if e > 0 else 'red' for e in errors]
        ax4.bar(range(len(errors)), errors, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('時間步')
        ax4.set_ylabel('預測誤差 (USD)')
        ax4.set_title('預測誤差時間序列', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 性能指標表
        ax5 = plt.subplot(3, 2, 5)
        ax5.axis('off')
        
        metrics_data = [
            ['指標', '數值', '評級'],
            ['', '', ''],
            ['MAE (USD)', f'${mae:.6f}', '✓' if mae < 0.2 else '✗'],
            ['MAPE (%)', f'{mape:.4f}%', '✓' if mape < 0.1 else '✗'],
            ['RMSE (USD)', f'${rmse:.6f}', ''],
            ['R² 分數', f'{r2:.4f}', '✓' if r2 > 0.90 else '✗'],
            ['方向準確度', f'{dir_acc:.2%}', '✓' if dir_acc > 0.65 else '✗'],
            ['', '', ''],
            ['測試樣本數', f'{len(y_true)}', ''],
            ['訓練狀態', 'v1.1 生產版', '✓'],
        ]
        
        table = ax5.table(cellText=metrics_data, loc='center', cellLoc='left',
                         colWidths=[0.35, 0.35, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # 格式化表頭
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 格式化行
        for i in range(1, len(metrics_data)):
            if metrics_data[i][0] == '':
                continue
            if i % 2 == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax5.set_title('性能指標', fontweight='bold', loc='left', fontsize=12)
        
        # 6. 價格變化方向對比
        ax6 = plt.subplot(3, 2, 6)
        true_dirs = np.diff(y_true, prepend=0) > 0
        pred_dirs = np.diff(y_pred, prepend=0) > 0
        
        x_pos = np.arange(len(true_dirs))
        width = 0.35
        
        ax6.bar(x_pos - width/2, true_dirs.astype(int), width, label='實際方向', alpha=0.7, color='#1f77b4')
        ax6.bar(x_pos + width/2, pred_dirs.astype(int), width, label='預測方向', alpha=0.7, color='#ff7f0e')
        
        ax6.set_ylabel('方向 (1=上升, 0=下降)')
        ax6.set_title('價格變化方向對比', fontweight='bold')
        ax6.set_ylim(-0.1, 1.2)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        
        plt.tight_layout()
        
        # 保存圖表
        output_dir = Path('results/visualizations')
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f'{self.symbol}_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 圖表已保存到: {output_path}")
        
        # 顯示圖表
        plt.show()
    
    @staticmethod
    def _calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """計算方向準確度"""
        # 計算實際價格變化方向
        y_true_diff = np.diff(y_true, prepend=0)
        y_true_dir = (y_true_diff > 0).astype(int)
        
        # 計算預測價格變化方向
        y_pred_diff = np.diff(y_pred, prepend=0)
        y_pred_dir = (y_pred_diff > 0).astype(int)
        
        # 計算準確度
        return accuracy_score(y_true_dir, y_pred_dir)


def main():
    parser = argparse.ArgumentParser(description='可視化模型預測結果')
    parser.add_argument('--symbol', type=str, default='SOL', help='幣種 (例: SOL, BTC)')
    parser.add_argument('--limit', type=int, default=200, help='載入的數據點數量')
    parser.add_argument('--model', type=str, default=None, help='模型路徑')
    parser.add_argument('--results', type=str, default=None, help='結果 JSON 路徑')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"模型預測可視化工具 - {args.symbol}")
    print("="*80 + "\n")
    
    visualizer = ModelVisualizer(args.symbol, args.model, args.results)
    
    print(f"正在獲取數據並進行預測...")
    pred_data = visualizer.fetch_and_predict(limit=args.limit)
    
    if pred_data is not None:
        print(f"✓ 成功獲取 {len(pred_data['y_true'])} 個預測樣本")
        print(f"\n正在生成可視化圖表...")
        visualizer.create_visualizations(pred_data)
    else:
        print(f"❌ 獲取數據失敗")


if __name__ == '__main__':
    main()
