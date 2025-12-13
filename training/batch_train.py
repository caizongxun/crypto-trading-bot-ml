#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量訓練腳本 - 一次訓練市值前 20 的幣種
支援斷點續訓、進度追蹤、自動 git 提交

用法:
  python training/batch_train.py                    # 訓練市值前 20 的幣種
  python training/batch_train.py --symbols SOL,BTC,ETH  # 訓練指定幣種
  python training/batch_train.py --symbols-file symbols.txt  # 從文件讀取
  python training/batch_train.py --resume SOL      # 從 SOL 繼續訓練
  python training/batch_train.py --epochs 150      # 自訂訓練輪數
  python training/batch_train.py --no-git          # 不自動提交
"""

import os
import sys
import io
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# 設定 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 市值前 20 的幣種（2025 年 12 月）
TOP_20_SYMBOLS = [
    'BTC',      # Bitcoin
    'ETH',      # Ethereum
    'BNB',      # Binance Coin
    'SOL',      # Solana
    'XRP',      # Ripple
    'DOGE',     # Dogecoin
    'ADA',      # Cardano
    'AVAX',     # Avalanche
    'LINK',     # Chainlink
    'MATIC',    # Polygon
    'ARB',      # Arbitrum
    'OP',       # Optimism
    'LDO',      # Lido
    'SUI',      # Sui
    'NEAR',     # Near
    'INJ',      # Injective
    'SEI',      # Sei
    'TON',      # Ton
    'FET',      # Fetch.ai
    'ICP',      # Internet Computer
]


class BatchTrainer:
    """批量訓練管理器"""
    
    def __init__(self, symbols: List[str], epochs: int = 200, auto_git: bool = True):
        self.symbols = symbols
        self.epochs = epochs
        self.auto_git = auto_git
        self.results = {}
        self.failed = {}
        self.start_time = datetime.now()
        self.log_file = Path('logs') / f'batch_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"批量訓練管理器")
        print(f"{'='*80}")
        print(f"幣種數量: {len(self.symbols)}")
        print(f"訓練輪數: {self.epochs}")
        print(f"自動提交: {self._yn(auto_git)}")
        print(f"日誌文件: {self.log_file}")
        print(f"\n開始訓練: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    def _yn(self, val: bool) -> str:
        return "✓ 是" if val else "✗ 否"
    
    def _log(self, msg: str):
        """記錄到文件和控制台"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        print(msg)
    
    def train_symbol(self, symbol: str, index: int, total: int) -> bool:
        """訓練單個幣種"""
        try:
            self._log(f"\n[{index}/{total}] 訓練 {symbol}...")
            self._log(f"  開始時間: {datetime.now().strftime('%H:%M:%S')}")
            
            # 檢查模型是否已存在
            model_path = f'models/saved/{symbol}_model.pth'
            if os.path.exists(model_path):
                self._log(f"  ⚠ 模型已存在，將覆蓋: {model_path}")
            
            # 執行訓練
            cmd = [
                'python',
                'training/train_lstm_v1.py',
                '--symbol', symbol,
                '--epochs', str(self.epochs)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 小時超時
            )
            
            if result.returncode == 0:
                # 訓練成功，提取結果
                results_file = f'results/{symbol}_results.json'
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        
                        mae = results.get('mae', 0)
                        mape = results.get('mape', 0)
                        acc = results.get('direction_accuracy', 0)
                        
                        self.results[symbol] = results
                        
                        self._log(f"  ✓ 訓練完成")
                        self._log(f"    MAE:  ${mae:.6f} {'✓' if mae < 0.2 else '✗'}")
                        self._log(f"    MAPE: {mape:.4f}% {'✓' if mape < 0.1 else '✗'}")
                        self._log(f"    方向準確度: {acc:.2%} {'✓' if acc > 0.65 else '✗'}")
                        self._log(f"  結束時間: {datetime.now().strftime('%H:%M:%S')}")
                        
                        # 自動提交
                        if self.auto_git:
                            self._git_commit(symbol, results)
                        
                        return True
                    
                    except json.JSONDecodeError:
                        self._log(f"  ✗ 無法解析結果 JSON")
                        self.failed[symbol] = "JSON 解析失敗"
                        return False
                else:
                    self._log(f"  ✗ 結果文件未生成")
                    self.failed[symbol] = "結果文件未生成"
                    return False
            else:
                error_msg = result.stderr[-500:] if result.stderr else "未知錯誤"
                self._log(f"  ✗ 訓練失敗")
                self._log(f"    錯誤: {error_msg}")
                self.failed[symbol] = error_msg
                return False
        
        except subprocess.TimeoutExpired:
            self._log(f"  ✗ 訓練超時 (> 1 小時)")
            self.failed[symbol] = "訓練超時"
            return False
        
        except Exception as e:
            self._log(f"  ✗ 訓練異常: {e}")
            self.failed[symbol] = str(e)
            return False
    
    def _git_commit(self, symbol: str, results: dict):
        """自動 git 提交"""
        try:
            mae = results.get('mae', 0)
            mape = results.get('mape', 0)
            acc = results.get('direction_accuracy', 0)
            
            msg = f"1-LSTM training: {symbol} model, MAE={mae:.4f}, MAPE={mape:.4f}%, Accuracy={acc:.2%}"
            
            subprocess.run(
                ['git', 'add', f'results/{symbol}_results.json', f'models/saved/{symbol}_model.pth'],
                capture_output=True,
                timeout=30
            )
            
            subprocess.run(
                ['git', 'commit', '-m', msg],
                capture_output=True,
                timeout=30
            )
            
            subprocess.run(
                ['git', 'push', 'origin', 'main'],
                capture_output=True,
                timeout=60
            )
            
            self._log(f"  ✓ 已提交到 GitHub")
        
        except Exception as e:
            self._log(f"  ⚠ Git 提交失敗: {e}")
    
    def train_all(self, resume_from: str = None):
        """訓練所有幣種"""
        start_idx = 0
        
        if resume_from:
            try:
                start_idx = self.symbols.index(resume_from)
                self._log(f"\n✓ 從 {resume_from} (第 {start_idx + 1}/{len(self.symbols)}) 繼續訓練\n")
            except ValueError:
                self._log(f"\n✗ 幣種 {resume_from} 不在列表中\n")
                return
        
        success_count = 0
        
        for idx in range(start_idx, len(self.symbols)):
            symbol = self.symbols[idx]
            try:
                if self.train_symbol(symbol, idx + 1, len(self.symbols)):
                    success_count += 1
            except KeyboardInterrupt:
                self._log(f"\n\n⚠ 用戶中斷訓練 (已完成 {success_count}/{idx} 個)")
                self._log(f"要繼續訓練，運行: python training/batch_train.py --resume {symbol}")
                break
            except Exception as e:
                self._log(f"\n✗ 異常錯誤: {e}")
                self.failed[symbol] = str(e)
        
        self.print_summary(success_count)
    
    def print_summary(self, success_count: int):
        """打印訓練總結"""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600
        
        self._log(f"\n\n{'='*80}")
        self._log("訓練總結")
        self._log(f"{'='*80}")
        self._log(f"\n總耗時: {hours:.1f} 小時 ({elapsed})")
        self._log(f"成功: {success_count}/{len(self.symbols)} 個")
        self._log(f"失敗: {len(self.failed)}/{len(self.symbols)} 個")
        
        if self.results:
            self._log(f"\n✓ 成功訓練的幣種:")
            
            # 按 MAE 排序
            sorted_results = sorted(
                self.results.items(),
                key=lambda x: x[1].get('mae', float('inf'))
            )
            
            for symbol, result in sorted_results:
                mae = result.get('mae', 0)
                mape = result.get('mape', 0)
                acc = result.get('direction_accuracy', 0)
                status = '✓' if (mae < 0.2 and mape < 0.1) else '⚠'
                self._log(f"  {status} {symbol:6s} | MAE=${mae:.4f} | MAPE={mape:.4f}% | Acc={acc:.2%}")
        
        if self.failed:
            self._log(f"\n✗ 失敗的幣種:")
            for symbol, error in self.failed.items():
                self._log(f"  ✗ {symbol}: {error[:50]}...")
        
        self._log(f"\n日誌文件: {self.log_file}")
        self._log(f"{'='*80}\n")
        
        # 如果有失敗，建議重試
        if self.failed:
            failed_symbols = ','.join(self.failed.keys())
            self._log(f"要重試失敗的幣種，運行:")
            self._log(f"  python training/batch_train.py --symbols {failed_symbols}")


def main():
    parser = argparse.ArgumentParser(
        description='批量訓練市值前 20 的幣種',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\n例子:
  python training/batch_train.py
  python training/batch_train.py --symbols SOL,BTC,ETH
  python training/batch_train.py --symbols-file symbols.txt
  python training/batch_train.py --resume SOL
  python training/batch_train.py --epochs 150
  python training/batch_train.py --no-git
        """
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='逗號分隔的幣種列表 (預設: 市值前 20)'
    )
    parser.add_argument(
        '--symbols-file',
        type=str,
        default=None,
        help='從文件讀取幣種列表 (每行一個)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='從指定幣種繼續訓練'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='每個模型的訓練輪數 (預設: 200)'
    )
    parser.add_argument(
        '--no-git',
        action='store_true',
        help='不自動提交到 GitHub'
    )
    
    args = parser.parse_args()
    
    # 決定要訓練的幣種
    if args.symbols_file:
        with open(args.symbols_file, 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = TOP_20_SYMBOLS
    
    # 創建訓練器並運行
    trainer = BatchTrainer(symbols, args.epochs, not args.no_git)
    trainer.train_all(resume_from=args.resume)


if __name__ == '__main__':
    main()
