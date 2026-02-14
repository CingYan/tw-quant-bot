#!/usr/bin/env python3
"""
台股量化交易系統 - 主程式
階段一：資料層實作
"""

import sys
import argparse
import logging
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.fetcher import TWSEDataFetcher
from data.processor import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/tw_quant.log')
    ]
)
logger = logging.getLogger(__name__)


def update_data(symbols=None, days=365):
    """更新股票資料"""
    logger.info("=== 開始更新資料 ===")
    
    fetcher = TWSEDataFetcher()
    fetcher.fetch_and_save(symbols=symbols, days=days)
    
    logger.info("=== 資料更新完成 ===")


def process_data(symbols=None):
    """處理股票資料"""
    logger.info("=== 開始處理資料 ===")
    
    if symbols is None:
        # 從配置讀取
        import json
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        symbols = config['data']['symbols']
    
    processor = DataProcessor()
    
    for symbol in symbols:
        logger.info(f"處理 {symbol}...")
        
        # 清洗資料
        count = processor.clean_data(symbol)
        
        if count > 0:
            # 計算技術指標
            processor.calculate_technical_indicators(symbol)
            
            # 匯出處理後的資料
            output_path = f"data/processed/{symbol}.json"
            processor.export_to_json(symbol, output_path)
        else:
            logger.warning(f"{symbol}: 無資料可處理")
    
    logger.info("=== 資料處理完成 ===")


def show_data(symbol, limit=10):
    """顯示股票資料"""
    processor = DataProcessor()
    data = processor.get_processed_data(symbol)
    
    if not data:
        logger.warning(f"{symbol}: 無資料")
        return
    
    logger.info(f"\n{symbol} 最近 {limit} 筆資料:")
    print(f"{'日期':<12} {'收盤':<8} {'成交量':<12} {'RSI':<8} {'MA5':<8} {'MA20':<8} {'量比':<8}")
    print("-" * 80)
    
    for row in data[-limit:]:
        date = row.get('date', 'N/A')
        close = row.get('close', 0)
        volume = row.get('volume', 0)
        rsi = row.get('rsi')
        ma5 = row.get('ma_5')
        ma20 = row.get('ma_20')
        vol_ratio = row.get('volume_ratio')
        
        print(f"{date:<12} {close:<8.2f} {volume:<12} "
              f"{rsi if rsi else 'N/A':<8} "
              f"{ma5 if ma5 else 'N/A':<8} "
              f"{ma20 if ma20 else 'N/A':<8} "
              f"{vol_ratio if vol_ratio else 'N/A':<8}")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='台股量化交易系統')
    
    subparsers = parser.add_subparsers(dest='command', help='可用指令')
    
    # update 指令
    update_parser = subparsers.add_parser('update', help='更新股票資料')
    update_parser.add_argument('--symbols', nargs='+', help='股票代碼列表')
    update_parser.add_argument('--days', type=int, default=365, help='回溯天數')
    
    # process 指令
    process_parser = subparsers.add_parser('process', help='處理股票資料')
    process_parser.add_argument('--symbols', nargs='+', help='股票代碼列表')
    
    # show 指令
    show_parser = subparsers.add_parser('show', help='顯示股票資料')
    show_parser.add_argument('symbol', help='股票代碼')
    show_parser.add_argument('--limit', type=int, default=10, help='顯示筆數')
    
    # all 指令（更新+處理）
    all_parser = subparsers.add_parser('all', help='更新並處理所有資料')
    all_parser.add_argument('--symbols', nargs='+', help='股票代碼列表')
    all_parser.add_argument('--days', type=int, default=365, help='回溯天數')
    
    args = parser.parse_args()
    
    if args.command == 'update':
        update_data(symbols=args.symbols, days=args.days)
    
    elif args.command == 'process':
        process_data(symbols=args.symbols)
    
    elif args.command == 'show':
        show_data(args.symbol, limit=args.limit)
    
    elif args.command == 'all':
        update_data(symbols=args.symbols, days=args.days)
        process_data(symbols=args.symbols)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # 創建必要的目錄
    Path("logs").mkdir(exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    main()
