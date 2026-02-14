#!/usr/bin/env python3
"""
監控系統主程式

整合：
- Telegram 通知
- Web UI 儀表板
- 自動化排程
"""

import argparse
import sys
from pathlib import Path


def start_telegram_notifier(target: str = None):
    """
    啟動 Telegram 通知器（測試模式）
    
    Args:
        target: 目標 chat_id（可選）
    """
    from src.monitor.telegram_bot import TelegramNotifier
    from src.data.fetcher_v2 import TWSEDataFetcherV2
    from src.ml.predictor import SignalGenerator, StockPredictor
    import pandas as pd
    
    print("📱 Telegram 通知器 - 測試模式")
    print("=" * 60)
    
    # 載入資料
    fetcher = TWSEDataFetcherV2()
    data = fetcher.get_stock_data('2603.TW')
    df = pd.DataFrame(data)
    
    # 載入模型
    predictor = StockPredictor.load('models/stock_predictor_2603.pkl')
    
    # 訊號生成
    signal_gen = SignalGenerator(
        predictor=predictor,
        ml_threshold=0.6,
        use_technical=True,
        use_ml=True
    )
    
    # 獲取最新訊號
    latest_signal = signal_gen.get_latest_signal(df)
    
    # 創建通知器（使用指定目標或預設）
    notifier = TelegramNotifier(chat_id=target) if target else TelegramNotifier()
    
    # 生成訊息
    message = notifier.send_signal_notification(latest_signal)
    
    print(message)
    print("\n" + "="*60)
    print("📱 請使用 OpenClaw message tool 發送此訊息到 Telegram")
    print(f"📍 目標：{notifier.chat_id}")
    print("\n💡 提示：")
    print("  - 羽燕鋒私訊：136149833（預設）")
    print("  - 柯姊敗家團：-1001068509881")
    print("  - Apex助理群：-5112325586")
    print("\n使用範例：")
    print("  python monitor.py telegram --target -1001068509881  # 發到柯姊敗家團")
    
    return message, notifier.chat_id


def start_web_ui(host: str = '0.0.0.0', port: int = 5000):
    """啟動 Web UI 儀表板"""
    from src.monitor.web_ui import start_web_server
    
    start_web_server(host=host, port=port, debug=False)


def start_scheduler():
    """啟動自動化排程器"""
    from src.monitor.scheduler import setup_default_schedule
    
    print("🔄 啟動自動化排程器...")
    print("=" * 60)
    
    scheduler = setup_default_schedule()
    scheduler.list_jobs()
    
    print("\n按 Ctrl+C 停止排程器...")
    scheduler.start(blocking=True)


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='台股量化交易系統 - 監控系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
範例：
  python monitor.py telegram        # 測試 Telegram 通知
  python monitor.py web              # 啟動 Web UI（http://localhost:5000）
  python monitor.py scheduler        # 啟動自動化排程
  python monitor.py web --host 0.0.0.0 --port 8080  # 自訂主機和埠號
        '''
    )
    
    parser.add_argument(
        'mode',
        choices=['telegram', 'web', 'scheduler'],
        help='運行模式'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Web UI 主機位址（預設：0.0.0.0）'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Web UI 埠號（預設：5000）'
    )
    
    parser.add_argument(
        '--target',
        default=None,
        help='Telegram 目標 chat_id（羽燕鋒私訊：136149833，柯姊敗家團：-1001068509881，Apex助理群：-5112325586）'
    )
    
    args = parser.parse_args()
    
    # 執行對應模式
    if args.mode == 'telegram':
        start_telegram_notifier(target=args.target)
    
    elif args.mode == 'web':
        start_web_ui(host=args.host, port=args.port)
    
    elif args.mode == 'scheduler':
        start_scheduler()


if __name__ == '__main__':
    main()
