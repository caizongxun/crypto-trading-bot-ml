#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
訓練結果可視化器
比較 v1、v2 和 v3 的訓練結果，統計指標

用法:
  python visualize_results.py                    # 比較所有幣種 v1 vs v2 vs v3
  python visualize_results.py --symbol SOL      # 統計 SOL 的訓練結果
  python visualize_results.py --compare         # 生成比較表格
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 設定中文字體
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class ResultsVisualizer:
    """結果可視化化等級"""
    
    def __init__(self):
        self.results_dir = Path('results')
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
        
        # 蒐取最新的結果
        self.v1_results = {}
        self.v2_results = {}
        self.v3_results = {}
        self._load_results()
    
    def _load_results(self):
        """載入所有結果檔案"""
        if not self.results_dir.exists():
            print(f"⚠ {self.results_dir} 不存在")
            return
        
        # 載入 V1
        for file in self.results_dir.glob('*_results.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                symbol = file.stem.replace('_results', '')
                self.v1_results[symbol] = data
            except:
                pass
        
        # 載入 V2
        for file in self.results_dir.glob('*_results_v2.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                symbol = file.stem.replace('_results_v2', '')
                self.v2_results[symbol] = data
            except:
                pass
        
        # 載入 V3
        for file in self.results_dir.glob('*_results_v3.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                symbol = file.stem.replace('_results_v3', '')
                self.v3_results[symbol] = data
            except:
                pass
        
        print(f"✓ 載入結果: {len(self.v1_results)} v1, {len(self.v2_results)} v2, {len(self.v3_results)} v3")
    
    def plot_single_symbol(self, symbol: str):
        """針對單個幣種的 v1 vs v2 vs v3 比較"""
        v1 = self.v1_results.get(symbol)
        v2 = self.v2_results.get(symbol)
        v3 = self.v3_results.get(symbol)
        
        if not v1 and not v2 and not v3:
            print(f"✗ {symbol} 沒有結果檔案")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{symbol} - V1 vs V2 vs V3 比較', fontsize=16, fontweight='bold')
        
        # MAE 比較
        ax = axes[0, 0]
        versions = []
        maes = []
        colors = []
        
        if v1:
            versions.append('V1')
            maes.append(v1.get('mae', 0))
            colors.append('#1f77b4')
        if v2:
            versions.append('V2')
            maes.append(v2.get('mae', 0))
            colors.append('#ff7f0e')
        if v3:
            versions.append('V3')
            maes.append(v3.get('mae', 0))
            colors.append('#2ca02c')
        
        ax.bar(versions, maes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('MAE (USD)', fontsize=12)
        ax.set_title('MAE 比較 (低者更佳)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(maes):
            ax.text(i, v, f'{v:.6f}', ha='center', va='bottom', fontweight='bold')
        
        # MAPE 比較
        ax = axes[0, 1]
        mapes = []
        versions = []
        colors = []
        
        if v1:
            versions.append('V1')
            mapes.append(v1.get('mape', 0) * 100)
            colors.append('#1f77b4')
        if v2:
            versions.append('V2')
            mapes.append(v2.get('mape', 0) * 100)
            colors.append('#ff7f0e')
        if v3:
            versions.append('V3')
            mapes.append(v3.get('mape', 0) * 100)
            colors.append('#2ca02c')
        
        ax.bar(versions, mapes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('MAPE 比較 (低者更佳)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(mapes):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 方向準確度 比較
        ax = axes[1, 0]
        accuracies = []
        versions = []
        colors = []
        
        if v1:
            versions.append('V1')
            accuracies.append(v1.get('direction_accuracy', 0) * 100)
            colors.append('#1f77b4')
        if v2:
            versions.append('V2')
            accuracies.append(v2.get('direction_accuracy', 0) * 100)
            colors.append('#ff7f0e')
        if v3:
            versions.append('V3')
            accuracies.append(v3.get('direction_accuracy', 0) * 100)
            colors.append('#2ca02c')
        
        ax.bar(versions, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('準確度 (%)', fontsize=12)
        ax.set_title('方向準確度比較 (高者更佳)', fontsize=12)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        for i, v in enumerate(accuracies):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # RMSE 比較
        ax = axes[1, 1]
        rmses = []
        versions = []
        colors = []
        
        if v1:
            versions.append('V1')
            rmses.append(v1.get('rmse', 0))
            colors.append('#1f77b4')
        if v2:
            versions.append('V2')
            rmses.append(v2.get('rmse', 0))
            colors.append('#ff7f0e')
        if v3:
            versions.append('V3')
            rmses.append(v3.get('rmse', 0))
            colors.append('#2ca02c')
        
        ax.bar(versions, rmses, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('RMSE (USD)', fontsize=12)
        ax.set_title('RMSE 比較 (低者更佳)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(rmses):
            ax.text(i, v, f'{v:.6f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_file = self.plots_dir / f'{symbol}_v1_vs_v2_vs_v3.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存: {output_file}")
        plt.close()
    
    def plot_all_comparison(self):
        """針對全部幣種的比較檔索"""
        all_symbols = set(self.v1_results.keys()) | set(self.v2_results.keys()) | set(self.v3_results.keys())
        if not all_symbols:
            print("✗ 沒有結果檔案")
            return
        
        all_symbols = sorted(all_symbols)
        
        # 技貼晦暗他了一乙形容載了昇障礎
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('所有幣種 - V1 vs V2 vs V3 比較', fontsize=16, fontweight='bold')
        
        # MAE 比較
        ax = axes[0, 0]
        v1_maes = []
        v2_maes = []
        v3_maes = []
        labels = []
        
        for symbol in all_symbols:
            labels.append(symbol)
            v1_maes.append(self.v1_results.get(symbol, {}).get('mae', 0))
            v2_maes.append(self.v2_results.get(symbol, {}).get('mae', 0))
            v3_maes.append(self.v3_results.get(symbol, {}).get('mae', 0))
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, v1_maes, width, label='V1', alpha=0.8, color='#1f77b4')
        ax.bar(x, v2_maes, width, label='V2', alpha=0.8, color='#ff7f0e')
        ax.bar(x + width, v3_maes, width, label='V3', alpha=0.8, color='#2ca02c')
        ax.set_xlabel('幣種', fontsize=12)
        ax.set_ylabel('MAE (USD)', fontsize=12)
        ax.set_title('MAE 比較', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # MAPE 比較
        ax = axes[0, 1]
        v1_mapes = []
        v2_mapes = []
        v3_mapes = []
        
        for symbol in all_symbols:
            v1_mapes.append(self.v1_results.get(symbol, {}).get('mape', 0) * 100)
            v2_mapes.append(self.v2_results.get(symbol, {}).get('mape', 0) * 100)
            v3_mapes.append(self.v3_results.get(symbol, {}).get('mape', 0) * 100)
        
        ax.bar(x - width, v1_mapes, width, label='V1', alpha=0.8, color='#1f77b4')
        ax.bar(x, v2_mapes, width, label='V2', alpha=0.8, color='#ff7f0e')
        ax.bar(x + width, v3_mapes, width, label='V3', alpha=0.8, color='#2ca02c')
        ax.set_xlabel('幣種', fontsize=12)
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title('MAPE 比較', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 方向準確度 比較
        ax = axes[1, 0]
        v1_accs = []
        v2_accs = []
        v3_accs = []
        
        for symbol in all_symbols:
            v1_accs.append(self.v1_results.get(symbol, {}).get('direction_accuracy', 0) * 100)
            v2_accs.append(self.v2_results.get(symbol, {}).get('direction_accuracy', 0) * 100)
            v3_accs.append(self.v3_results.get(symbol, {}).get('direction_accuracy', 0) * 100)
        
        ax.bar(x - width, v1_accs, width, label='V1', alpha=0.8, color='#1f77b4')
        ax.bar(x, v2_accs, width, label='V2', alpha=0.8, color='#ff7f0e')
        ax.bar(x + width, v3_accs, width, label='V3', alpha=0.8, color='#2ca02c')
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('幣種', fontsize=12)
        ax.set_ylabel('準確度 (%)', fontsize=12)
        ax.set_title('方向準確度比較', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # RMSE 比較
        ax = axes[1, 1]
        v1_rmses = []
        v2_rmses = []
        v3_rmses = []
        
        for symbol in all_symbols:
            v1_rmses.append(self.v1_results.get(symbol, {}).get('rmse', 0))
            v2_rmses.append(self.v2_results.get(symbol, {}).get('rmse', 0))
            v3_rmses.append(self.v3_results.get(symbol, {}).get('rmse', 0))
        
        ax.bar(x - width, v1_rmses, width, label='V1', alpha=0.8, color='#1f77b4')
        ax.bar(x, v2_rmses, width, label='V2', alpha=0.8, color='#ff7f0e')
        ax.bar(x + width, v3_rmses, width, label='V3', alpha=0.8, color='#2ca02c')
        ax.set_xlabel('幣種', fontsize=12)
        ax.set_ylabel('RMSE (USD)', fontsize=12)
        ax.set_title('RMSE 比較', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.plots_dir / 'all_symbols_comparison_v1_v2_v3.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存: {output_file}")
        plt.close()
    
    def generate_summary_table(self):
        """產生总结表格"""
        all_symbols = sorted(set(self.v1_results.keys()) | set(self.v2_results.keys()) | set(self.v3_results.keys()))
        
        data = []
        for symbol in all_symbols:
            v1 = self.v1_results.get(symbol, {})
            v2 = self.v2_results.get(symbol, {})
            v3 = self.v3_results.get(symbol, {})
            
            row = {
                '幣種': symbol,
                'V1 MAE': f"{v1.get('mae', 0):.6f}",
                'V2 MAE': f"{v2.get('mae', 0):.6f}",
                'V3 MAE': f"{v3.get('mae', 0):.6f}",
                'V1 準確': f"{v1.get('direction_accuracy', 0)*100:.2f}%",
                'V2 準確': f"{v2.get('direction_accuracy', 0)*100:.2f}%",
                'V3 準確': f"{v3.get('direction_accuracy', 0)*100:.2f}%",
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        output_file = self.plots_dir / 'comparison_summary_v1_v2_v3.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 已保存: {output_file}")
        
        # 也打印表格
        print("\n" + "="*120)
        print("訓練結果總結")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
    
    def run(self, symbol: str = None, compare: bool = False):
        """運行可視化"""
        print(f"\n正在產生圖表...")
        
        if symbol:
            self.plot_single_symbol(symbol)
        else:
            self.plot_all_comparison()
        
        self.generate_summary_table()
        
        print(f"\n✓ 所有圖表已保存到 plots/ 目錄")
        print(f"\n引總:")
        print(f"  - V1 結果: {len(self.v1_results)} 個幣種")
        print(f"  - V2 結果: {len(self.v2_results)} 個幣種")
        print(f"  - V3 結果: {len(self.v3_results)} 個幣種")
        print(f"  - 圖表位置: plots/")
        print(f"  - 總结表: plots/comparison_summary_v1_v2_v3.csv")


def main():
    parser = argparse.ArgumentParser(description='訓練結果可視化')
    parser.add_argument('--symbol', type=str, default=None, help='指定幣種')
    parser.add_argument('--compare', action='store_true', help='生成比較表')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer()
    visualizer.run(symbol=args.symbol, compare=args.compare)


if __name__ == '__main__':
    main()
