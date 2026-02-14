#!/usr/bin/env python3
"""
交易機器人（整合 OpenClaw message tool）

自動執行交易並推送訊息到 Telegram
"""

import time
from datetime import datetime
from auto_trading import AutoTrader


class TradingBot(AutoTrader):
    """交易機器人（支援 Telegram 推送）"""
    
    def send_to_telegram(self, message: str):
        """
        發送訊息到 Telegram
        
        Args:
            message: 訊息內容
        """
        print(f"\n📱 [Telegram] 發送訊息到 {self.target_chat}")
        print("="*60)
        print(message)
        print("="*60)
        print("\n💡 實際運作時，此訊息會通過 OpenClaw message tool 自動發送\n")
    
    def run_once_with_telegram(self):
        """執行一次檢查並發送到 Telegram"""
        signal_msg, trade_msg, summary_msg = self.run_once()
        
        # 如果有交易，發送三則訊息
        if trade_msg:
            # 1. 訊號
            self.send_to_telegram(signal_msg)
            time.sleep(1)
            
            # 2. 交易記錄
            self.send_to_telegram(trade_msg)
            time.sleep(1)
            
            # 3. 帳戶摘要
            self.send_to_telegram(summary_msg)
        
        return bool(trade_msg)
    
    def start_with_telegram(self, interval_minutes: int = 30):
        """啟動交易系統（支援 Telegram）"""
        print("=" * 60)
        print("🤖 假設性交易系統啟動（Telegram 整合版）")
        print("=" * 60)
        print(f"💰 初始資金：{self.account.initial_capital:,.0f} TWD")
        print(f"📊 交易標的：{self.symbol}")
        print(f"⏰ 檢查間隔：每 {interval_minutes} 分鐘")
        print(f"📱 訊息推送：柯姊敗家團（{self.target_chat}）")
        print("=" * 60)
        print()
        
        # 啟動時發送帳戶初始狀態
        current_price = self.get_current_price()
        init_msg = f"""
🚀 **假設性交易系統已啟動**

**初始資金：** {self.account.initial_capital:,.0f} TWD
**交易標的：** {self.symbol}
**當前價格：** {current_price:.2f} TWD
**檢查間隔：** 每 {interval_minutes} 分鐘

---

_⏰ 九點開始運作，根據訊號自動交易_
_🤖 台股量化交易系統 v1.0_
""".strip()
        
        self.send_to_telegram(init_msg)
        
        print("\n按 Ctrl+C 停止...\n")
        
        try:
            while True:
                has_trade = self.run_once_with_telegram()
                
                if not has_trade:
                    print(f"⏰ 下次檢查：{interval_minutes} 分鐘後\n")
                
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            print("\n⏹️ 交易系統已停止")
            
            # 發送最終摘要
            current_price = self.get_current_price()
            final_msg = f"""
⏹️ **交易系統已停止**

{self.account.format_summary_message({self.symbol: current_price})}

---

_感謝使用台股量化交易系統_
""".strip()
            
            self.send_to_telegram(final_msg)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='交易機器人')
    parser.add_argument('--interval', type=int, default=30, help='檢查間隔（分鐘）')
    parser.add_argument('--test', action='store_true', help='測試模式')
    
    args = parser.parse_args()
    
    bot = TradingBot(target_chat="-1001068509881")  # 柯姊敗家團
    
    if args.test:
        print("🧪 測試模式\n")
        bot.run_once_with_telegram()
    else:
        bot.start_with_telegram(interval_minutes=args.interval)
