#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速可視化 - 直接使用現有模型生成圖表
无需重新訓練，立即生成高質量图表

稨粘用法：
  python training/quick_visualize.py --symbol SOL
  python training/quick_visualize.py --symbol BTC --limit 500
  python training/quick_visualize.py --symbol ETH --limit 300 --show
"""

import os
import sys
import io
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)

# 設定 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設定圖表風格
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
sns.set_style("darkgrid")

# 顏色定義
COLOR_TRUE = '#1f77b4'      # 藍色
COLOR_PRED = '#ff7f0e'      # 橙色
COLOR_ERROR_POS = '#2ca02c' # 綠色
COLOR_ERROR_NEG = '#d62728' # 紅色
COLOR_GRID = '#666666'      # 灰色


class QuickVisualizer:
    """快速可視化類 - 無需重新訓練"""
    
    def __init__(self, symbol: str, model_dir: str = 'models/saved'):
        self.symbol = symbol
        self.model_dir = model_dir
        self.model_path = f'{model_dir}/{symbol}_model.pth'
        self.results_path = f'results/{symbol}_results.json'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print(f"快速圖表生成工具 - {symbol}")
        print(f"{'='*80}")
        print(f"設備: {self.device}")
        print(f"模型路徑: {self.model_path}")
        print(f"結果路徑: {self.results_path}")
    
    def check_model_exists(self) -> bool:
        """檢查模型是否存在"""
        if not os.path.exists(self.model_path):
            print(f"\n❌ 模型不存在: {self.model_path}")
            print(f"   請先訓練模型: python training/train_lstm_v1.py --symbol {self.symbol}")
            return False
        print(f"✓ 模型已找到: {self.model_path}")
        return True
    
    def load_results(self) -> dict:
        """載入訓練結果 JSON"""
        if not os.path.exists(self.results_path):
            print(f"⚠ 結果文件不存在: {self.results_path}")
            return None
        
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"✓ 已載入結果文件")
            return results
        except Exception as e:
            print(f"❌ 載入結果失敗: {e}")
            return None
    
    def fetch_and_predict(self, limit: int = 200) -> dict:
        """獲取數據並進行預測"""
        try:
            import ccxt
            import yaml
            
            print(f"\n[1/3] 獲取數據...")
            print(f"      正在從 Binance 拉取 {limit} 根 K 線...")
            
            # 載入配置
            with open('training/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 動態載入訓練模塊
            from training.train_lstm_v1 import (
                LSTMModel,
                add_technical_indicators,
                prepare_sequences
            )
            
            # 獲取數據
            exchange = ccxt.binance()
            trading_pair = f"{self.symbol}/USDT"
            ohlcv = exchange.fetch_ohlcv(trading_pair, '1h', limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"      ✓ 已獲取 {len(df)} 根 K 線")
            
            # 添加技術指標
            print(f"[2/3] 特徵提取...")
            df = add_technical_indicators(df)
            print(f"      ✓ 已添加 {len(df.columns) - 6} 個技術指標")
            
            # 特徵提取
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
            X = df[feature_cols].values
            y = df['close'].values
            
            # 正規化
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # 確保特徵數為 44
            if X_scaled.shape[1] > 44:
                X_scaled = X_scaled[:, :44]
            elif X_scaled.shape[1] < 44:
                padding = np.zeros((X_scaled.shape[0], 44 - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
            
            # 準備序列
            X_seq, y_seq = prepare_sequences(
                X_scaled,
                y_scaled,
                config['training']['lookback_window']
            )
            
            # 載入模型
            print(f"[3/3] 模型推理...")
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
            
            print(f"      ✓ 已完成 {len(y_preds)} 個預測")
            
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
    
    def create_charts(self, pred_data: dict):
        """創建 6 個圖表"""
        if pred_data is None:
            print("❌ 沒有預測數據")
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
        
        print(f"\n📊 性能指標:")
        print(f"   MAE:  ${mae:.6f} {'✓' if mae < 0.2 else '✗'}")
        print(f"   MAPE: {mape:.4f}% {'✓' if mape < 0.1 else '✗'}")
        print(f"   RMSE: ${rmse:.6f}")
        print(f"   R²:   {r2:.4f} {'✓' if r2 > 0.90 else '✗'}")
        print(f"   方向準確度: {dir_acc:.2%} {'✓' if dir_acc > 0.65 else '✗'}")
        
        # 創建圖表
        print(f"\n正在生成圖表...")
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f'{self.symbol} LSTM 預測準確度分析 - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        # 1. 價格預測對比
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(timestamps, y_true, label='實際價格', color=COLOR_TRUE, linewidth=2, alpha=0.8)
        ax1.plot(timestamps, y_pred, label='預測價格', color=COLOR_PRED, linewidth=1.5, alpha=0.7, linestyle='--')
        ax1.fill_between(timestamps, y_true, y_pred, alpha=0.2, color=COLOR_GRID)
        ax1.set_xlabel('時間')
        ax1.set_ylabel('價格 (USDT)')
        ax1.set_title(f'價格預測對比 (MAE: ${mae:.4f})', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 誤差分布
        ax2 = plt.subplot(3, 2, 2)
        errors = np.abs(y_true - y_pred)
        ax2.hist(errors, bins=30, color=COLOR_ERROR_POS, alpha=0.7, edgecolor='black')
        ax2.axvline(mae, color=COLOR_ERROR_NEG, linestyle='--', linewidth=2, label=f'平均誤差: ${mae:.4f}')
        ax2.set_xlabel('絕對誤差 (USD)')
        ax2.set_ylabel('頻率')
        ax2.set_title('誤差分布', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 散點圖
        ax3 = plt.subplot(3, 2, 3)
        ax3.scatter(y_true, y_pred, alpha=0.5, s=20, color=COLOR_ERROR_NEG)
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
        colors = [COLOR_ERROR_POS if e > 0 else COLOR_ERROR_NEG for e in errors]
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
        
        table = ax5.table(
            cellText=metrics_data,
            loc='center',
            cellLoc='left',
            colWidths=[0.35, 0.35, 0.2]
        )
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
        
        # 6. 方向對比
        ax6 = plt.subplot(3, 2, 6)
        true_dirs = np.diff(y_true, prepend=0) > 0
        pred_dirs = np.diff(y_pred, prepend=0) > 0
        
        x_pos = np.arange(len(true_dirs))
        width = 0.35
        
        ax6.bar(x_pos - width/2, true_dirs.astype(int), width, label='實際方向', alpha=0.7, color=COLOR_TRUE)
        ax6.bar(x_pos + width/2, pred_dirs.astype(int), width, label='預測方向', alpha=0.7, color=COLOR_PRED)
        
        ax6.set_ylabel('方向 (1=上升, 0=下降)')
        ax6.set_title('價格變化方向對比', fontweight='bold')
        ax6.set_ylim(-0.1, 1.2)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_xticks(x_pos[::max(1, len(x_pos)//10)])
        
        plt.tight_layout()
        
        # 保存圖表
        output_dir = Path('results/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f'{self.symbol}_predictions_{timestamp}.png'
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 圖表已保存到: {output_path}")
        
        return str(output_path)
    
    @staticmethod
    def _calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """計算方向準確度"""
        true_dirs = np.diff(y_true, prepend=0) > 0
        pred_dirs = np.diff(y_pred, prepend=0) > 0
        return np.mean(true_dirs == pred_dirs)


def main():
    parser = argparse.ArgumentParser(
        description='快速圖表生成 - 直接使用已有模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\n範例用法:
  python training/quick_visualize.py --symbol SOL
  python training/quick_visualize.py --symbol BTC --limit 500
  python training/quick_visualize.py --symbol ETH --limit 300 --show
        """
    )
    parser.add_argument('--symbol', type=str, default='SOL', help='幣種 (例: SOL, BTC, ETH)')
    parser.add_argument('--limit', type=int, default=200, help='載入的數據點數量 (預設: 200)')
    parser.add_argument('--model-dir', type=str, default='models/saved', help='模型目錄')
    parser.add_argument('--show', action='store_true', help='顯示圖表')
    
    args = parser.parse_args()
    
    # 建立可視化器
    visualizer = QuickVisualizer(args.symbol, args.model_dir)
    
    # 檢查模型
    if not visualizer.check_model_exists():
        sys.exit(1)
    
    # 預測並生成圖表
    pred_data = visualizer.fetch_and_predict(limit=args.limit)
    
    if pred_data is not None:
        chart_path = visualizer.create_charts(pred_data)
        
        # 顯示圖表
        if args.show:
            print(f"\n正在顯示圖表...")
            plt.show()
        else:
            print(f"\n💡 提示: 添加 --show 參數以顯示圖表")
            print(f"   python training/quick_visualize.py --symbol {args.symbol} --show")
    else:
        print(f"❌ 生成圖表失敗")
        sys.exit(1)
    
    print(f"\n✨ 完成！")


if __name__ == '__main__':
    main()
