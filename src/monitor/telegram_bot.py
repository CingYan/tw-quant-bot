"""
Telegram 通知模組

整合 OpenClaw 的 message tool，實現：
- 交易訊號推送
- 績效報告
- 價格監控
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime


class TelegramNotifier:
    """Telegram 訊號通知器（整合 OpenClaw）"""
    
    def __init__(self, chat_id: str = "136149833"):
        """
        初始化通知器
        
        Args:
            chat_id: Telegram 目標 ID（預設：羽燕鋒私訊）
                    - 羽燕鋒私訊：136149833
                    - 柯姊敗家團：-1001068509881
                    - Apex助理群：-5112325586
        """
        self.chat_id = chat_id
        self.notification_enabled = True
    
    def send_signal_notification(self, signal_data: Dict[str, Any]) -> str:
        """
        發送交易訊號通知
        
        Args:
            signal_data: 訊號資料字典
        
        Returns:
            格式化的訊息文字
        """
        # 格式化訊號訊息
        message = self._format_signal_message(signal_data)
        
        # 返回訊息（由 OpenClaw message tool 發送）
        return message
    
    def send_daily_report(self, report_data: Dict[str, Any]) -> str:
        """
        發送每日績效報告
        
        Args:
            report_data: 報告資料
        
        Returns:
            格式化的報告文字
        """
        message = self._format_daily_report(report_data)
        return message
    
    def send_price_alert(self, symbol: str, price: float, change_pct: float) -> str:
        """
        發送價格警報
        
        Args:
            symbol: 股票代碼
            price: 當前價格
            change_pct: 漲跌幅
        
        Returns:
            格式化的警報文字
        """
        direction = "🔴 下跌" if change_pct < 0 else "🟢 上漲"
        
        message = f"""
⚠️ **價格警報**

股票：{symbol}
當前價格：{price:.2f} TWD
漲跌幅：{direction} {abs(change_pct):.2%}

時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()
        
        return message
    
    @staticmethod
    def _format_signal_message(signal: Dict[str, Any]) -> str:
        """格式化交易訊號訊息"""
        # 訊號圖示
        signal_emoji = {
            1: "🟢",
            -1: "🔴",
            0: "⚪"
        }.get(signal.get('signal', 0), "⚪")
        
        # 推薦文字
        recommendation = signal.get('recommendation', '⚪ 觀望')
        
        # 格式化訊息
        message = f"""
{signal_emoji} **台股交易訊號**

**股票：** 台積電 2330.TW
**日期：** {signal.get('date', 'N/A')}
**收盤價：** {signal.get('close', 0):.2f} TWD

---

**綜合訊號：** {recommendation}

**技術分析分數：** {signal.get('technical_score', 0)}
**ML 上漲機率：** {signal.get('ml_proba_up', 0):.2%}

**技術指標：**
• RSI(14): {signal.get('rsi_14', 0):.2f}
• MA5/MA20: {'金叉 ✓' if signal.get('ma_5_20_cross') else '死叉 ✗'}

---

_💡 雙重確認策略：技術分析 + ML 預測_
""".strip()
        
        return message
    
    @staticmethod
    def _format_daily_report(report: Dict[str, Any]) -> str:
        """格式化每日報告"""
        message = f"""
📊 **每日績效報告**

**日期：** {report.get('date', 'N/A')}

**訊號統計：**
• 買入訊號：{report.get('buy_signals', 0)} 次
• 賣出訊號：{report.get('sell_signals', 0)} 次
• 觀望：{report.get('hold_signals', 0)} 次

**準確率：**
• 買入準確率：{report.get('buy_accuracy', 0):.2%}
• 賣出準確率：{report.get('sell_accuracy', 0):.2%}

**模型效能：**
• 測試集準確率：{report.get('test_accuracy', 0):.2%}

---

_📈 台股量化交易系統 v0.3_
""".strip()
        
        return message
    
    def enable_notifications(self):
        """啟用通知"""
        self.notification_enabled = True
    
    def disable_notifications(self):
        """停用通知"""
        self.notification_enabled = False


# ==================== 命令行工具 ====================

def send_latest_signal_notification():
    """發送最新訊號通知（命令行工具）"""
    from ..data.fetcher_v2 import TWSEDataFetcherV2
    from ..ml.predictor import SignalGenerator, StockPredictor
    import pandas as pd
    
    # 載入資料
    fetcher = TWSEDataFetcherV2()
    data = fetcher.get_stock_data('2330.TW')
    df = pd.DataFrame(data)
    
    # 載入模型
    predictor = StockPredictor.load('models/stock_predictor_2330.pkl')
    
    # 訊號生成
    signal_gen = SignalGenerator(
        predictor=predictor,
        ml_threshold=0.6,
        use_technical=True,
        use_ml=True
    )
    
    # 獲取最新訊號
    latest_signal = signal_gen.get_latest_signal(df)
    
    # 創建通知器
    notifier = TelegramNotifier()
    
    # 生成訊息
    message = notifier.send_signal_notification(latest_signal)
    
    print(message)
    print("\n" + "="*60)
    print("📱 請使用 OpenClaw message tool 發送此訊息到 Telegram")
    print(f"📍 目標群組：{notifier.chat_id}")
    
    return message


if __name__ == '__main__':
    send_latest_signal_notification()
