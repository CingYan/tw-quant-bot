#!/usr/bin/env python3
"""
台積電假設性交易
"""
import time
from datetime import datetime
import yfinance as yf
from paper_trading import PaperTradingAccount, format_trade_message
from src.ml.predictor import SignalGenerator, StockPredictor

class QuickTrader:
    def __init__(self, symbol="2330.TW"):
        self.symbol = symbol
        self.account = PaperTradingAccount(initial_capital=1_000_000)
        
        print("📥 載入 ML 模型...")
        try:
            self.predictor = StockPredictor.load(f'models/stock_predictor_2330.pkl')
            self.signal_gen = SignalGenerator(
                predictor=self.predictor,
                ml_threshold=0.6,
                use_technical=True,
                use_ml=True
            )
            print("✅ ML 模型已載入")
        except Exception as e:
            print(f"⚠️ 使用純技術分析模式: {e}")
            self.predictor = None
            self.signal_gen = None
    
    def get_signal(self):
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period='3mo')
        
        close = df['Close'].iloc[-1]
        
        # 技術指標
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 評分
        score = 0
        if ma5 > ma20: score += 1
        if rsi < 70: score += 1
        if rsi > 30: score += 1
        
        # 訊號
        if score >= 2 and ma5 > ma20 and rsi < 60:
            signal = 1  # 買入
            recommendation = "買入"
        elif rsi > 70 or ma5 < ma20:
            signal = -1  # 賣出
            recommendation = "賣出"
        else:
            signal = 0  # 觀望
            recommendation = "觀望"
        
        return {
            'signal': signal,
            'recommendation': recommendation,
            'close': close,
            'rsi': rsi,
            'ma5': ma5,
            'ma20': ma20,
            'ma_cross': 'golden' if ma5 > ma20 else 'death',
            'score': score
        }
    
    def run(self, interval_minutes=30):
        print("="*60)
        print(f"🤖 台積電假設性交易系統啟動")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 初始資金：1,000,000 TWD")
        print(f"📊 標的：{self.symbol}")
        print("="*60)
        
        last_signal = None
        
        while True:
            try:
                signal = self.get_signal()
                now = datetime.now().strftime('%H:%M:%S')
                
                emoji = {'買入': '🟢', '賣出': '🔴', '觀望': '⚪'}[signal['recommendation']]
                print(f"[{now}] {emoji} {signal['recommendation']} | 價格: {signal['close']:.2f} | RSI: {signal['rsi']:.1f}")
                
                # 執行交易
                if signal['signal'] == 1 and last_signal != 1:
                    available = self.account.cash * 0.8
                    shares = int(available / signal['close'] / 1000) * 1000
                    if shares >= 1000:
                        result = self.account.buy(self.symbol, signal['close'], shares)
                        if result['status'] == 'success':
                            print(f"    ✅ 買入 {shares} 股 @ {signal['close']:.2f}")
                        last_signal = 1
                
                elif signal['signal'] == -1 and last_signal != -1:
                    if self.symbol in self.account.positions:
                        shares = self.account.positions[self.symbol]['shares']
                        result = self.account.sell(self.symbol, signal['close'], shares)
                        if result['status'] == 'success':
                            print(f"    ✅ 賣出 {shares} 股 @ {signal['close']:.2f}")
                        last_signal = -1
                
                # 帳戶狀態
                total_value = self.account.get_total_value({self.symbol: signal['close']})
                pnl = total_value - self.account.initial_capital
                pnl_pct = (pnl / self.account.initial_capital) * 100
                print(f"    💰 總值: {total_value:,.0f} TWD | P&L: {pnl:+,.0f} ({pnl_pct:+.2f}%)")
                
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n⏹️ 停止")
                break
            except Exception as e:
                print(f"⚠️ 錯誤: {e}")
                time.sleep(60)

if __name__ == '__main__':
    trader = QuickTrader("2330.TW")
    trader.run(interval_minutes=30)
