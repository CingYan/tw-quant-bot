"""
自動化排程模組

定時執行：
- 資料更新
- 模型預測
- 訊號推送
"""

import schedule
import time
from typing import Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoScheduler:
    """自動化排程器"""
    
    def __init__(self):
        self.jobs = []
        self.running = False
    
    def add_daily_update(self, hour: int = 9, minute: int = 0):
        """
        添加每日資料更新任務
        
        Args:
            hour: 小時（0-23）
            minute: 分鐘（0-59）
        """
        def update_task():
            logger.info("🔄 開始每日資料更新...")
            from ..data.fetcher_v2 import TWSEDataFetcherV2
            
            fetcher = TWSEDataFetcherV2()
            fetcher.fetch_and_save(symbols=['2330.TW'], period='1y')
            logger.info("✅ 資料更新完成")
        
        job = schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(update_task)
        self.jobs.append(job)
        logger.info(f"✅ 已排程每日資料更新：{hour:02d}:{minute:02d}")
    
    def add_signal_check(self, interval_minutes: int = 30):
        """
        添加訊號檢查任務
        
        Args:
            interval_minutes: 檢查間隔（分鐘）
        """
        def signal_task():
            logger.info("🔍 檢查交易訊號...")
            from ..data.fetcher_v2 import TWSEDataFetcherV2
            from ..ml.predictor import SignalGenerator, StockPredictor
            from .telegram_bot import TelegramNotifier
            import pandas as pd
            
            # 載入資料
            fetcher = TWSEDataFetcherV2()
            data = fetcher.get_stock_data('2330.TW')
            df = pd.DataFrame(data)
            
            # 載入模型
            try:
                predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
            except:
                logger.warning("⚠️ 模型未找到，跳過訊號檢查")
                return
            
            # 訊號生成
            signal_gen = SignalGenerator(
                predictor=predictor,
                ml_threshold=0.6,
                use_technical=True,
                use_ml=True
            )
            
            latest_signal = signal_gen.get_latest_signal(df)
            
            # 如果有買入/賣出訊號，則推送通知
            if latest_signal['signal'] != 0:
                notifier = TelegramNotifier()
                message = notifier.send_signal_notification(latest_signal)
                logger.info(f"📱 發送訊號通知：{latest_signal['recommendation']}")
                print(message)
            else:
                logger.info("⚪ 當前訊號：觀望")
        
        job = schedule.every(interval_minutes).minutes.do(signal_task)
        self.jobs.append(job)
        logger.info(f"✅ 已排程訊號檢查：每 {interval_minutes} 分鐘")
    
    def add_daily_report(self, hour: int = 18, minute: int = 0):
        """
        添加每日報告任務
        
        Args:
            hour: 小時（0-23）
            minute: 分鐘（0-59）
        """
        def report_task():
            logger.info("📊 生成每日報告...")
            from .telegram_bot import TelegramNotifier
            
            # 生成報告資料（簡化版）
            report_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'buy_signals': 1,
                'sell_signals': 0,
                'hold_signals': 29,
                'buy_accuracy': 0.8462,
                'sell_accuracy': 1.0000,
                'test_accuracy': 0.4250
            }
            
            notifier = TelegramNotifier()
            message = notifier.send_daily_report(report_data)
            logger.info("✅ 每日報告已生成")
            print(message)
        
        job = schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(report_task)
        self.jobs.append(job)
        logger.info(f"✅ 已排程每日報告：{hour:02d}:{minute:02d}")
    
    def add_custom_task(self, func: Callable, interval_minutes: int = 60):
        """
        添加自訂任務
        
        Args:
            func: 任務函數
            interval_minutes: 執行間隔（分鐘）
        """
        job = schedule.every(interval_minutes).minutes.do(func)
        self.jobs.append(job)
        logger.info(f"✅ 已排程自訂任務：每 {interval_minutes} 分鐘")
    
    def start(self, blocking: bool = True):
        """
        啟動排程器
        
        Args:
            blocking: 是否阻塞執行（True = 持續執行，False = 後台執行）
        """
        self.running = True
        logger.info("🚀 自動化排程器已啟動")
        logger.info(f"📅 已排程任務數：{len(self.jobs)}")
        
        if blocking:
            try:
                while self.running:
                    schedule.run_pending()
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("⏸️ 排程器已停止（用戶中斷）")
                self.running = False
        else:
            logger.info("🔄 非阻塞模式：請手動調用 run_pending()")
    
    def stop(self):
        """停止排程器"""
        self.running = False
        logger.info("⏹️ 排程器已停止")
    
    def list_jobs(self):
        """列出所有任務"""
        logger.info(f"\n📋 排程任務列表（共 {len(self.jobs)} 個）：")
        for idx, job in enumerate(self.jobs, 1):
            logger.info(f"  {idx}. {job}")
    
    @staticmethod
    def run_pending():
        """執行待處理任務（非阻塞模式使用）"""
        schedule.run_pending()


# ==================== 預設排程配置 ====================

def setup_default_schedule():
    """設定預設排程"""
    scheduler = AutoScheduler()
    
    # 每日資料更新（09:00）
    scheduler.add_daily_update(hour=9, minute=0)
    
    # 訊號檢查（每 30 分鐘）
    scheduler.add_signal_check(interval_minutes=30)
    
    # 每日報告（18:00）
    scheduler.add_daily_report(hour=18, minute=0)
    
    return scheduler


# ==================== 測試程式碼 ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='自動化排程器')
    parser.add_argument('--mode', choices=['default', 'test'], default='default',
                        help='運行模式（default=預設排程, test=測試模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # 測試模式：立即執行一次訊號檢查
        logger.info("🧪 測試模式：執行訊號檢查...")
        
        from ..data.fetcher_v2 import TWSEDataFetcherV2
        from ..ml.predictor import SignalGenerator, StockPredictor
        from .telegram_bot import TelegramNotifier
        import pandas as pd
        
        fetcher = TWSEDataFetcherV2()
        data = fetcher.get_stock_data('2330.TW')
        df = pd.DataFrame(data)
        
        predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
        signal_gen = SignalGenerator(predictor=predictor, ml_threshold=0.6)
        latest_signal = signal_gen.get_latest_signal(df)
        
        notifier = TelegramNotifier()
        message = notifier.send_signal_notification(latest_signal)
        
        print("\n" + "="*60)
        print(message)
        print("="*60)
    
    else:
        # 預設模式：啟動排程器
        logger.info("🚀 啟動預設排程...")
        scheduler = setup_default_schedule()
        scheduler.list_jobs()
        
        print("\n按 Ctrl+C 停止排程器...")
        scheduler.start(blocking=True)
