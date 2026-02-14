#!/usr/bin/env python3
"""
自動交易系統（假設性）

九點開始運作，根據訊號自動交易
所有訊息發送到柯姊敗家團
"""

import time
from datetime import datetime
from paper_trading import PaperTradingAccount, format_trade_message
import yfinance as yf
from src.ml.predictor import SignalGenerator, StockPredictor
import pandas as pd


class AutoTrader:
    """自動交易系統"""
    
    def __init__(self, target_chat: str = "-1001068509881"):
        """
        初始化自動交易系統
        
        Args:
            target_chat: 目標群組 ID（預設：柯姊敗家團）
        """
        self.account = PaperTradingAccount(initial_capital=1_000_000)
        self.target_chat = target_chat
        self.symbol = "2409.TW"
        
        # 載入模型
        print("📥 載入 ML 模型...")
        self.predictor = StockPredictor.load('models/stock_predictor_2409.pkl')
        self.signal_gen = SignalGenerator(
            predictor=self.predictor,
            ml_threshold=0.6,
            use_technical=True,
            use_ml=True
        )
        print("✅ ML 模型已載入\n")
        
        self.last_signal = None
    
    def get_latest_signal(self):
        """獲取最新訊號"""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period='1y')
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        return self.signal_gen.get_latest_signal(df)
    
    def get_current_price(self) -> float:
        """獲取當前價格（使用最新收盤價）"""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period='5d')
        return float(df['Close'].iloc[-1])
    
    def execute_signal(self, signal: dict, current_price: float) -> str:
        """
        執行訊號
        
        Args:
            signal: 訊號資料
            current_price: 當前價格
        
        Returns:
            執行訊息
        """
        signal_value = signal['signal']
        
        # 買入訊號
        if signal_value == 1:
            # 計算可買股數（使用 80% 現金）
            available_cash = self.account.cash * 0.8
            shares = int(available_cash / current_price / 1000) * 1000  # 整千股
            
            if shares >= 1000:
                result = self.account.buy(self.symbol, current_price, shares)
                
                if result['status'] == 'success':
                    return format_trade_message(result['trade'])
                else:
                    return f"❌ 買入失敗：{result['reason']}"
            else:
                return f"⚠️ 資金不足，無法買入（可用現金：{self.account.cash:,.0f}）"
        
        # 賣出訊號
        elif signal_value == -1:
            if self.symbol in self.account.positions:
                shares = self.account.positions[self.symbol]['shares']
                result = self.account.sell(self.symbol, current_price, shares)
                
                if result['status'] == 'success':
                    return format_trade_message(result['trade'])
                else:
                    return f"❌ 賣出失敗：{result['reason']}"
            else:
                return "⚠️ 無持倉，無法賣出"
        
        # 觀望
        else:
            return None
    
    def format_signal_message(self, signal: dict, current_price: float) -> str:
        """格式化訊號訊息"""
        signal_emoji = {
            1: "🟢",
            -1: "🔴",
            0: "⚪"
        }.get(signal['signal'], "⚪")
        
        message = f"""
{signal_emoji} **台股交易訊號**

**股票：** {self.symbol}
**時間：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**當前價格：** {current_price:.2f} TWD

---

**綜合訊號：** {signal['recommendation']}

**技術分析分數：** {signal['technical_score']}
**ML 上漲機率：** {signal['ml_proba_up']:.2%}

**技術指標：**
• RSI(14): {signal['rsi_14']:.2f}
• MA5/MA20: {'金叉 ✓' if signal['ma_5_20_cross'] else '死叉 ✗'}
""".strip()
        
        return message
    
    def run_once(self) -> tuple:
        """
        執行一次檢查
        
        Returns:
            (訊號訊息, 交易訊息, 帳戶摘要訊息)
        """
        print(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] 檢查訊號...")
        
        # 獲取最新訊號
        signal = self.get_latest_signal()
        current_price = self.get_current_price()
        
        # 格式化訊號訊息
        signal_msg = self.format_signal_message(signal, current_price)
        
        # 檢查訊號是否改變
        if signal['signal'] != 0 and signal['signal'] != self.last_signal:
            print(f"  ⚡ 新訊號：{signal['recommendation']}")
            
            # 執行交易
            trade_msg = self.execute_signal(signal, current_price)
            
            # 更新最後訊號
            self.last_signal = signal['signal']
            
            # 獲取帳戶摘要
            summary_msg = self.account.format_summary_message({self.symbol: current_price})
            
            return signal_msg, trade_msg, summary_msg
        else:
            print(f"  {signal['recommendation']}")
            return signal_msg, None, None
    
    def start(self, interval_minutes: int = 30):
        """
        啟動自動交易系統
        
        Args:
            interval_minutes: 檢查間隔（分鐘）
        """
        print("=" * 60)
        print("🤖 自動交易系統啟動")
        print("=" * 60)
        print(f"📍 初始資金：{self.account.initial_capital:,.0f} TWD")
        print(f"📍 交易標的：{self.symbol}")
        print(f"📍 檢查間隔：每 {interval_minutes} 分鐘")
        print(f"📍 訊息推送：柯姊敗家團")
        print("=" * 60)
        print()
        
        print("按 Ctrl+C 停止...\n")
        
        try:
            while True:
                signal_msg, trade_msg, summary_msg = self.run_once()
                
                # 這裡應該使用 OpenClaw message tool 發送
                # 但在測試階段，我們先打印出來
                if trade_msg:
                    print("\n" + "="*60)
                    print(signal_msg)
                    print("\n" + "-"*60)
                    print(trade_msg)
                    print("\n" + "-"*60)
                    print(summary_msg)
                    print("="*60 + "\n")
                
                # 等待下次檢查
                print(f"⏰ 下次檢查：{interval_minutes} 分鐘後\n")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            print("\n⏹️ 自動交易系統已停止")
            
            # 最終摘要
            current_price = self.get_current_price()
            final_summary = self.account.format_summary_message({self.symbol: current_price})
            print("\n" + "="*60)
            print("📊 最終帳戶摘要")
            print("="*60)
            print(final_summary)
            print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='自動交易系統')
    parser.add_argument('--interval', type=int, default=30, help='檢查間隔（分鐘）')
    parser.add_argument('--test', action='store_true', help='測試模式（立即執行一次）')
    
    args = parser.parse_args()
    
    trader = AutoTrader()
    
    if args.test:
        # 測試模式：立即執行一次
        print("🧪 測試模式：執行一次檢查\n")
        signal_msg, trade_msg, summary_msg = trader.run_once()
        
        print("\n" + "="*60)
        print(signal_msg)
        if trade_msg:
            print("\n" + "-"*60)
            print(trade_msg)
            print("\n" + "-"*60)
            print(summary_msg)
        print("="*60)
    else:
        # 正常模式：持續運作
        trader.start(interval_minutes=args.interval)
