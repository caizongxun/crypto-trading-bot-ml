#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
訓練結果可視化器
比較 v1、v2、v3、v4 和 v5 的訓練結果，統計指標

V5 是方向分類，所以會顯示：Accuracy / Precision / Recall / F1

用法:
  python visualize_results.py                    # 比較所有幣種 v1-v5
  python visualize_results.py --symbol SOL      # 統計 SOL 的訓練結果
"""

import os
import json
import argparse
from pathlib import Path
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.v4_results = {}
        self.v5_results = {}
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
        
        # 載入 V4
        for file in self.results_dir.glob('*_results_v4.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                symbol = file.stem.replace('_results_v4', '')
                self.v4_results[symbol] = data
            except:
                pass
        
        # 載入 V5
        for file in self.results_dir.glob('*_results_v5.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                symbol = file.stem.replace('_results_v5', '')
                self.v5_results[symbol] = data
            except:
                pass
        
        print(f"✓ 載入結果: {len(self.v1_results)} v1, {len(self.v2_results)} v2, {len(self.v3_results)} v3, {len(self.v4_results)} v4, {len(self.v5_results)} v5")
    
    def plot_single_symbol(self, symbol: str):
        """針對單個幣種的比較"""
        v1 = self.v1_results.get(symbol)
        v2 = self.v2_results.get(symbol)
        v3 = self.v3_results.get(symbol)
        v4 = self.v4_results.get(symbol)
        v5 = self.v5_results.get(symbol)
        
        if not any([v1, v2, v3, v4, v5]):
            print(f"✗ {symbol} 沒有結果檔案")
            return
        
        # V1-V4 使用回歸指標，V5 使用分類指標
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(f'{symbol} - V1-V4 (回歸) vs V5 (分類) 比較', fontsize=16, fontweight='bold')
        
        # MAE 比較 (V1-V4)
        ax = axes[0, 0]
        versions, maes, colors = [], [], []
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
        if v4:
            versions.append('V4')
            maes.append(v4.get('mae', 0))
            colors.append('#d62728')
        
        ax.bar(versions, maes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('MAE (USD)', fontsize=12)
        ax.set_title('MAE 比較 - V1~V4回歸 (低者更佳)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(maes):
            ax.text(i, v, f'{v:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 方向準確度 - V1-V4 回歸 vs V5 分類
        ax = axes[0, 1]
        versions_acc, accs_acc, colors_acc = [], [], []
        if v1:
            versions_acc.append('V1\n(Regr)')
            accs_acc.append(v1.get('direction_accuracy', 0)*100)
            colors_acc.append('#1f77b4')
        if v2:
            versions_acc.append('V2\n(Regr)')
            accs_acc.append(v2.get('direction_accuracy', 0)*100)
            colors_acc.append('#ff7f0e')
        if v3:
            versions_acc.append('V3\n(Regr)')
            accs_acc.append(v3.get('direction_accuracy', 0)*100)
            colors_acc.append('#2ca02c')
        if v4:
            versions_acc.append('V4\n(Regr)')
            accs_acc.append(v4.get('direction_accuracy', 0)*100)
            colors_acc.append('#d62728')
        if v5:
            versions_acc.append('V5\n(Class)')
            accs_acc.append(v5.get('direction_accuracy', 0)*100)
            colors_acc.append('#9467bd')
        
        ax.bar(versions_acc, accs_acc, color=colors_acc, alpha=0.7, edgecolor='black')
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('方向準確度 (高者更佳)', fontsize=12)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        for i, v in enumerate(accs_acc):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # V5 分類指標 (Precision / Recall / F1)
        ax = axes[1, 0]
        if v5:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            values = [
                v5.get('direction_accuracy', 0)*100,
                v5.get('direction_precision', 0)*100,
                v5.get('direction_recall', 0)*100,
                v5.get('direction_f1', 0)*100
            ]
            ax.bar(metrics, values, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'], alpha=0.7, edgecolor='black')
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_title('V5 方向分類指標', fontsize=12)
            ax.set_ylim([0, 100])
            ax.grid(axis='y', alpha=0.3)
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'V5 組簽沒有', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title('V5 方向分類指標', fontsize=12)
        
        # 比較推玩
        ax = axes[1, 1]
        ax.axis('off')
        comparison_text = "V1-V4 回歸沼解抐\nV5 方向分類方案\n\n"
        
        if v1:
            comparison_text += f"V1: {v1.get('mae', 0):.6f} MAE, {v1.get('direction_accuracy', 0)*100:.2f}%\n"
        if v2:
            comparison_text += f"V2: {v2.get('mae', 0):.6f} MAE, {v2.get('direction_accuracy', 0)*100:.2f}%\n"
        if v3:
            comparison_text += f"V3: {v3.get('mae', 0):.6f} MAE, {v3.get('direction_accuracy', 0)*100:.2f}%\n"
        if v4:
            comparison_text += f"V4: {v4.get('mae', 0):.6f} MAE, {v4.get('direction_accuracy', 0)*100:.2f}%\n"
        if v5:
            comparison_text += f"V5: {v5.get('direction_accuracy', 0)*100:.2f}% Acc (Classification)\n"
        
        ax.text(0.1, 0.9, comparison_text, 
               ha='left', va='top', fontsize=11, family='monospace',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_file = self.plots_dir / f'{symbol}_v1_v2_v3_v4_v5.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存: {output_file}")
        plt.close()
    
    def plot_all_comparison(self):
        """針對全部幣種的比較"""
        all_symbols = set(self.v1_results.keys()) | set(self.v2_results.keys()) | set(self.v3_results.keys()) | set(self.v4_results.keys()) | set(self.v5_results.keys())
        if not all_symbols:
            print("✗ 沒有結果檔案")
            return
        
        all_symbols = sorted(all_symbols)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('所有幣種 - V1~V4 (回歸) vs V5 (分類) 比較', fontsize=16, fontweight='bold')
        
        x = np.arange(len(all_symbols))
        width = 0.16
        
        # MAE 比較
        ax = axes[0, 0]
        v1_maes = [self.v1_results.get(s, {}).get('mae', 0) for s in all_symbols]
        v2_maes = [self.v2_results.get(s, {}).get('mae', 0) for s in all_symbols]
        v3_maes = [self.v3_results.get(s, {}).get('mae', 0) for s in all_symbols]
        v4_maes = [self.v4_results.get(s, {}).get('mae', 0) for s in all_symbols]
        
        ax.bar(x - 1.5*width, v1_maes, width, label='V1', color='#1f77b4', alpha=0.8)
        ax.bar(x - 0.5*width, v2_maes, width, label='V2', color='#ff7f0e', alpha=0.8)
        ax.bar(x + 0.5*width, v3_maes, width, label='V3', color='#2ca02c', alpha=0.8)
        ax.bar(x + 1.5*width, v4_maes, width, label='V4', color='#d62728', alpha=0.8)
        ax.set_xlabel('幣種')
        ax.set_ylabel('MAE (USD)')
        ax.set_title('MAE 比較')
        ax.set_xticks(x)
        ax.set_xticklabels(all_symbols, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 方向準確度
        ax = axes[0, 1]
        v1_accs = [self.v1_results.get(s, {}).get('direction_accuracy', 0)*100 for s in all_symbols]
        v2_accs = [self.v2_results.get(s, {}).get('direction_accuracy', 0)*100 for s in all_symbols]
        v3_accs = [self.v3_results.get(s, {}).get('direction_accuracy', 0)*100 for s in all_symbols]
        v4_accs = [self.v4_results.get(s, {}).get('direction_accuracy', 0)*100 for s in all_symbols]
        v5_accs = [self.v5_results.get(s, {}).get('direction_accuracy', 0)*100 for s in all_symbols]
        
        ax.bar(x - 2*width, v1_accs, width, label='V1 (Regr)', color='#1f77b4', alpha=0.8)
        ax.bar(x - width, v2_accs, width, label='V2 (Regr)', color='#ff7f0e', alpha=0.8)
        ax.bar(x, v3_accs, width, label='V3 (Regr)', color='#2ca02c', alpha=0.8)
        ax.bar(x + width, v4_accs, width, label='V4 (Regr)', color='#d62728', alpha=0.8)
        ax.bar(x + 2*width, v5_accs, width, label='V5 (Class)', color='#9467bd', alpha=0.8)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('幣種')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('方向準確度比較')
        ax.set_xticks(x)
        ax.set_xticklabels(all_symbols, rotation=45)
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # V5 Precision
        ax = axes[1, 0]
        v5_prec = [self.v5_results.get(s, {}).get('direction_precision', 0)*100 for s in all_symbols]
        ax.bar(x, v5_prec, width=0.6, label='V5 Precision', color='#8c564b', alpha=0.8)
        ax.set_xlabel('幣種')
        ax.set_ylabel('Precision (%)')
        ax.set_title('V5 精準率 (Precision)')
        ax.set_xticks(x)
        ax.set_xticklabels(all_symbols, rotation=45)
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # V5 Recall & F1
        ax = axes[1, 1]
        v5_recall = [self.v5_results.get(s, {}).get('direction_recall', 0)*100 for s in all_symbols]
        v5_f1 = [self.v5_results.get(s, {}).get('direction_f1', 0)*100 for s in all_symbols]
        
        ax.bar(x - width/2, v5_recall, width=width*0.9, label='Recall', color='#e377c2', alpha=0.8)
        ax.bar(x + width/2, v5_f1, width=width*0.9, label='F1', color='#7f7f7f', alpha=0.8)
        ax.set_xlabel('幣種')
        ax.set_ylabel('Score (%)')
        ax.set_title('V5 Recall & F1')
        ax.set_xticks(x)
        ax.set_xticklabels(all_symbols, rotation=45)
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.plots_dir / 'all_symbols_comparison_v1_v2_v3_v4_v5.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存: {output_file}")
        plt.close()
    
    def generate_summary_table(self):
        """產生总结表格"""
        all_symbols = sorted(set(self.v1_results.keys()) | set(self.v2_results.keys()) | set(self.v3_results.keys()) | set(self.v4_results.keys()) | set(self.v5_results.keys()))
        
        data = []
        for symbol in all_symbols:
            v1 = self.v1_results.get(symbol, {})
            v2 = self.v2_results.get(symbol, {})
            v3 = self.v3_results.get(symbol, {})
            v4 = self.v4_results.get(symbol, {})
            v5 = self.v5_results.get(symbol, {})
            
            row = {
                '幣種': symbol,
                'V1 MAE': f"{v1.get('mae', 0):.6f}",
                'V2 MAE': f"{v2.get('mae', 0):.6f}",
                'V3 MAE': f"{v3.get('mae', 0):.6f}",
                'V4 MAE': f"{v4.get('mae', 0):.6f}",
                'V1 Acc%': f"{v1.get('direction_accuracy', 0)*100:.2f}%",
                'V2 Acc%': f"{v2.get('direction_accuracy', 0)*100:.2f}%",
                'V3 Acc%': f"{v3.get('direction_accuracy', 0)*100:.2f}%",
                'V4 Acc%': f"{v4.get('direction_accuracy', 0)*100:.2f}%",
                'V5 Acc%': f"{v5.get('direction_accuracy', 0)*100:.2f}%",
                'V5 F1': f"{v5.get('direction_f1', 0)*100:.2f}%",
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        output_file = self.plots_dir / 'comparison_summary_v1_v2_v3_v4_v5.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 已保存: {output_file}")
        
        print("\n" + "="*180)
        print("訓練結果總結")
        print("="*180)
        print(df.to_string(index=False))
        print("="*180)
    
    def run(self, symbol: str = None):
        """運行可視化"""
        print(f"\n正在產生圖表...")
        
        if symbol:
            self.plot_single_symbol(symbol)
        else:
            self.plot_all_comparison()
        
        self.generate_summary_table()
        
        print(f"\n✓ 所有圖表已保存到 plots/ 目錄")
        print(f"\n引總:")
        print(f"  - V1-V4 結果: 回歸模式")
        print(f"  - V5 結果: 分類模式")
        print(f"  - V1 結果: {len(self.v1_results)} 個幣種")
        print(f"  - V2 結果: {len(self.v2_results)} 個幣種")
        print(f"  - V3 結果: {len(self.v3_results)} 個幣種")
        print(f"  - V4 結果: {len(self.v4_results)} 個幣種")
        print(f"  - V5 結果: {len(self.v5_results)} 個幣種")
        print(f"  - 圖表位置: plots/")
        print(f"  - 總结表: plots/comparison_summary_v1_v2_v3_v4_v5.csv")


def main():
    parser = argparse.ArgumentParser(description='訓練結果可視化')
    parser.add_argument('--symbol', type=str, default=None, help='指定幣種')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer()
    visualizer.run(symbol=args.symbol)


if __name__ == '__main__':
    main()
