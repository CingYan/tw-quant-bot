#!/usr/bin/env python3
"""
當沖交易系統 v2.0

改進版：
1. 當日沖銷（開盤買、收盤賣）
2. 多股票篩選
3. 停損機制（-3%）
4. 避免追高（不買已漲 > 5% 的股票）
5. 量價確認
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import List, Dict, Optional, Tuple
from paper_trading import PaperTradingAccount, format_trade_message


class DayTradingBot:
    """當沖交易機器人"""
    
    # 候選股票池（高流動性股票）
    STOCK_POOL = [
        "2330.TW",  # 台積電
        "2317.TW",  # 鴻海
        "2454.TW",  # 聯發科
        "2308.TW",  # 台達電
        "2881.TW",  # 富邦金
        "2882.TW",  # 國泰金
        "2412.TW",  # 中華電
        "3008.TW",  # 大立光
        "2303.TW",  # 聯電
        "2409.TW",  # 友達
    ]
    
    def __init__(self, initial_capital: float = 1_000_000):
        """初始化"""
        self.account = PaperTradingAccount(initial_capital=initial_capital)
        self.stop_loss_pct = -0.03  # -3% 停損
        self.max_gain_entry = 0.05  # 不買已漲超過 5% 的股票
        self.min_volume = 1000  # 最小成交量（張）
        self.today_trades = []
        
    def get_stock_data(self, symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """獲取股票數據"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                return None
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"⚠️ 無法獲取 {symbol} 數據: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """計算技術指標"""
        if len(df) < 20:
            return None
            
        close = df['close']
        volume = df['volume']
        
        # 均線
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_hist = macd - signal
        
        # 今日漲跌幅
        today_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        
        # 量能變化
        vol_ma5 = volume.rolling(5).mean()
        vol_ratio = volume.iloc[-1] / vol_ma5.iloc[-1] if vol_ma5.iloc[-1] > 0 else 1
        
        return {
            'close': close.iloc[-1],
            'ma5': ma5.iloc[-1],
            'ma10': ma10.iloc[-1],
            'ma20': ma20.iloc[-1],
            'rsi': rsi.iloc[-1],
            'macd_hist': macd_hist.iloc[-1],
            'today_change': today_change,
            'volume': volume.iloc[-1],
            'vol_ratio': vol_ratio,
            'ma_bullish': close.iloc[-1] > ma5.iloc[-1] > ma10.iloc[-1],
            'macd_bullish': macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2],
        }
    
    def score_stock(self, indicators: Dict) -> Tuple[float, str]:
        """
        評分股票（0-100）
        
        Returns:
            (分數, 原因說明)
        """
        score = 50  # 基礎分
        reasons = []
        
        # 1. 均線排列 (+20/-20)
        if indicators['ma_bullish']:
            score += 20
            reasons.append("✅ 均線多頭排列")
        else:
            score -= 10
            reasons.append("⚠️ 均線未多頭")
        
        # 2. RSI 位置 (+15/-15)
        rsi = indicators['rsi']
        if 40 <= rsi <= 60:
            score += 15
            reasons.append(f"✅ RSI 適中 ({rsi:.1f})")
        elif rsi < 30:
            score += 10
            reasons.append(f"✅ RSI 超賣 ({rsi:.1f})")
        elif rsi > 70:
            score -= 15
            reasons.append(f"⚠️ RSI 超買 ({rsi:.1f})")
        
        # 3. MACD (+15/-10)
        if indicators['macd_bullish']:
            score += 15
            reasons.append("✅ MACD 金叉/向上")
        else:
            score -= 10
            reasons.append("⚠️ MACD 偏空")
        
        # 4. 今日漲幅限制 (-20 如果已漲太多)
        change = indicators['today_change']
        if change > self.max_gain_entry:
            score -= 30
            reasons.append(f"❌ 已漲 {change:.1%}，避免追高")
        elif change < -0.03:
            score -= 10
            reasons.append(f"⚠️ 已跌 {change:.1%}")
        
        # 5. 量能 (+10/-5)
        if indicators['vol_ratio'] > 1.2:
            score += 10
            reasons.append(f"✅ 量能放大 ({indicators['vol_ratio']:.1f}x)")
        elif indicators['vol_ratio'] < 0.5:
            score -= 5
            reasons.append("⚠️ 量能萎縮")
        
        return max(0, min(100, score)), "\n".join(reasons)
    
    def select_best_stock(self) -> Optional[Dict]:
        """選擇最佳股票"""
        candidates = []
        
        print("🔍 分析候選股票...")
        for symbol in self.STOCK_POOL:
            df = self.get_stock_data(symbol, period="1mo")  # 使用1個月數據
            if df is None:
                continue
                
            indicators = self.calculate_indicators(df)
            if indicators is None:
                continue
            
            score, reasons = self.score_stock(indicators)
            
            candidates.append({
                'symbol': symbol,
                'score': score,
                'reasons': reasons,
                'indicators': indicators
            })
            
            print(f"  {symbol}: {score:.0f}分")
        
        if not candidates:
            return None
        
        # 選擇分數最高且 >= 60 的股票
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        if best['score'] >= 60:
            return best
        else:
            print(f"⚠️ 最高分 {best['score']:.0f} 未達 60 分門檻，今日不操作")
            return None
    
    def morning_analysis(self) -> str:
        """
        早盤分析（8:50 調用）
        
        Returns:
            分析報告
        """
        report = f"""
📊 **當沖早盤分析**
📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

"""
        candidates = []
        
        for symbol in self.STOCK_POOL:
            df = self.get_stock_data(symbol, period="1mo")
            if df is None:
                continue
                
            indicators = self.calculate_indicators(df)
            if indicators is None:
                continue
            
            score, reasons = self.score_stock(indicators)
            candidates.append({
                'symbol': symbol,
                'score': score,
                'price': indicators['close'],
                'reasons': reasons
            })
        
        # 排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        report += "**候選股票排名：**\n\n"
        for i, c in enumerate(candidates[:5], 1):
            report += f"{i}. **{c['symbol']}** - {c['score']:.0f}分 (${c['price']:.2f})\n"
        
        if candidates and candidates[0]['score'] >= 60:
            best = candidates[0]
            report += f"\n---\n\n🎯 **推薦標的：{best['symbol']}**\n\n{best['reasons']}"
        else:
            report += "\n---\n\n⚠️ 今日無符合條件的標的，建議觀望"
        
        return report.strip()
    
    def execute_buy(self) -> Optional[str]:
        """
        執行買入（9:00 調用）
        
        Returns:
            交易訊息
        """
        best = self.select_best_stock()
        
        if best is None:
            return "⚠️ 今日無符合條件的標的，不進場"
        
        symbol = best['symbol']
        price = best['indicators']['close']
        
        # 計算可買股數（使用 80% 資金）
        available = self.account.cash * 0.8
        shares = int(available / price / 1000) * 1000
        
        if shares < 1000:
            return f"⚠️ 資金不足，無法買入 {symbol}"
        
        result = self.account.buy(symbol, price, shares)
        
        if result['status'] == 'success':
            self.today_trades.append({
                'symbol': symbol,
                'buy_price': price,
                'shares': shares,
                'buy_time': datetime.now()
            })
            
            return f"""
🟢 **當沖買入**

**股票：** {symbol}
**價格：** {price:.2f} TWD
**數量：** {shares:,} 股
**金額：** {price * shares:,.0f} TWD

**選股理由：**
{best['reasons']}

**停損價：** {price * (1 + self.stop_loss_pct):.2f} (-3%)
""".strip()
        else:
            return f"❌ 買入失敗：{result['reason']}"
    
    def check_stop_loss(self) -> Optional[str]:
        """檢查是否觸發停損"""
        messages = []
        
        for trade in self.today_trades:
            symbol = trade['symbol']
            buy_price = trade['buy_price']
            
            df = self.get_stock_data(symbol, period="1d")
            if df is None:
                continue
            
            current_price = df['close'].iloc[-1]
            pnl_pct = (current_price - buy_price) / buy_price
            
            if pnl_pct <= self.stop_loss_pct:
                # 觸發停損
                shares = trade['shares']
                result = self.account.sell(symbol, current_price, shares)
                
                if result['status'] == 'success':
                    messages.append(f"""
🔴 **停損賣出**

**股票：** {symbol}
**買入價：** {buy_price:.2f}
**賣出價：** {current_price:.2f}
**損益：** {pnl_pct:.2%}
""".strip())
                    self.today_trades.remove(trade)
        
        return "\n\n".join(messages) if messages else None
    
    def execute_close(self) -> str:
        """
        收盤賣出（13:25 調用）
        
        Returns:
            交易訊息
        """
        if not self.today_trades:
            return "📋 今日無持倉，無需平倉"
        
        messages = []
        total_pnl = 0
        
        for trade in self.today_trades[:]:  # 複製列表以避免修改時出錯
            symbol = trade['symbol']
            buy_price = trade['buy_price']
            shares = trade['shares']
            
            df = self.get_stock_data(symbol, period="1d")
            if df is None:
                continue
            
            current_price = df['close'].iloc[-1]
            result = self.account.sell(symbol, current_price, shares)
            
            if result['status'] == 'success':
                pnl = (current_price - buy_price) * shares
                pnl_pct = (current_price - buy_price) / buy_price
                total_pnl += pnl
                
                emoji = "🟢" if pnl >= 0 else "🔴"
                messages.append(f"""
{emoji} **平倉：{symbol}**
買入：{buy_price:.2f} → 賣出：{current_price:.2f}
損益：{pnl:+,.0f} TWD ({pnl_pct:+.2%})
""".strip())
                
                self.today_trades.remove(trade)
        
        # 總結
        summary = f"""
📊 **當沖日結**
📅 {datetime.now().strftime('%Y-%m-%d')}

---

{"".join(messages)}

---

**今日總損益：** {total_pnl:+,.0f} TWD
**帳戶餘額：** {self.account.cash:,.0f} TWD
""".strip()
        
        return summary


# 測試
if __name__ == "__main__":
    bot = DayTradingBot()
    
    print("=== 早盤分析 ===")
    print(bot.morning_analysis())
    
    print("\n=== 模擬買入 ===")
    print(bot.execute_buy())
    
    print("\n=== 模擬平倉 ===")
    print(bot.execute_close())
