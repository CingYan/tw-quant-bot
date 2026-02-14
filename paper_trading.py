#!/usr/bin/env python3
"""
假設性交易測試（Paper Trading）

初始資金：100 萬 TWD
交易標的：長榮 2603.TW
訊號推送：柯姊敗家團（-1001068509881）
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class PaperTradingAccount:
    """假設性交易帳戶"""
    
    def __init__(self, initial_capital: float = 1_000_000):
        """
        初始化交易帳戶
        
        Args:
            initial_capital: 初始資金（預設：100 萬）
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float}}
        self.trade_history = []
        self.equity_curve = []
        
        # 交易成本
        self.commission_rate = 0.001425  # 手續費 0.1425%
        self.tax_rate = 0.003  # 證交稅 0.3%（賣出時）
        
        self.log_file = Path('paper_trading_log.json')
        self._load_state()
    
    def _load_state(self):
        """載入交易狀態"""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.cash = state.get('cash', self.initial_capital)
                self.positions = state.get('positions', {})
                self.trade_history = state.get('trade_history', [])
                self.equity_curve = state.get('equity_curve', [])
    
    def _save_state(self):
        """儲存交易狀態"""
        state = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'equity_curve': self.equity_curve,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def buy(self, symbol: str, price: float, shares: int) -> Dict:
        """
        買入股票
        
        Args:
            symbol: 股票代碼
            price: 買入價格
            shares: 股數
        
        Returns:
            交易記錄
        """
        # 計算交易成本
        cost = price * shares
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        # 檢查資金是否足夠
        if total_cost > self.cash:
            return {
                'status': 'failed',
                'reason': f'資金不足（需要 {total_cost:,.0f}，可用 {self.cash:,.0f}）'
            }
        
        # 執行買入
        self.cash -= total_cost
        
        if symbol in self.positions:
            # 加碼
            old_shares = self.positions[symbol]['shares']
            old_avg_price = self.positions[symbol]['avg_price']
            new_shares = old_shares + shares
            new_avg_price = (old_avg_price * old_shares + price * shares) / new_shares
            
            self.positions[symbol] = {
                'shares': new_shares,
                'avg_price': new_avg_price
            }
        else:
            # 新倉位
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price
            }
        
        # 記錄交易
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': 'buy',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'cost': cost,
            'commission': commission,
            'total_cost': total_cost,
            'remaining_cash': self.cash
        }
        
        self.trade_history.append(trade)
        self._save_state()
        
        return {
            'status': 'success',
            'trade': trade
        }
    
    def sell(self, symbol: str, price: float, shares: int) -> Dict:
        """
        賣出股票
        
        Args:
            symbol: 股票代碼
            price: 賣出價格
            shares: 股數
        
        Returns:
            交易記錄
        """
        # 檢查是否持有
        if symbol not in self.positions:
            return {
                'status': 'failed',
                'reason': f'未持有 {symbol}'
            }
        
        # 檢查股數是否足夠
        if self.positions[symbol]['shares'] < shares:
            return {
                'status': 'failed',
                'reason': f'持股不足（持有 {self.positions[symbol]["shares"]}，欲賣出 {shares}）'
            }
        
        # 計算交易成本
        revenue = price * shares
        commission = revenue * self.commission_rate
        tax = revenue * self.tax_rate
        net_revenue = revenue - commission - tax
        
        # 計算損益
        avg_price = self.positions[symbol]['avg_price']
        profit = (price - avg_price) * shares
        profit_pct = (price / avg_price - 1) * 100
        
        # 執行賣出
        self.cash += net_revenue
        self.positions[symbol]['shares'] -= shares
        
        # 如果賣光了就移除持倉
        if self.positions[symbol]['shares'] == 0:
            del self.positions[symbol]
        
        # 記錄交易
        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': 'sell',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'revenue': revenue,
            'commission': commission,
            'tax': tax,
            'net_revenue': net_revenue,
            'remaining_cash': self.cash,
            'avg_cost': avg_price,
            'profit': profit,
            'profit_pct': profit_pct
        }
        
        self.trade_history.append(trade)
        self._save_state()
        
        return {
            'status': 'success',
            'trade': trade
        }
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        計算總資產價值
        
        Args:
            current_prices: {symbol: price} 當前價格
        
        Returns:
            總資產價值
        """
        holdings_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                holdings_value += position['shares'] * current_prices[symbol]
        
        return self.cash + holdings_value
    
    def get_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        獲取帳戶摘要
        
        Args:
            current_prices: 當前價格（可選）
        
        Returns:
            帳戶摘要
        """
        if current_prices is None:
            current_prices = {}
        
        total_value = self.get_portfolio_value(current_prices)
        total_profit = total_value - self.initial_capital
        total_profit_pct = (total_value / self.initial_capital - 1) * 100
        
        holdings = []
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['avg_price'])
            market_value = position['shares'] * current_price
            cost = position['shares'] * position['avg_price']
            profit = market_value - cost
            profit_pct = (current_price / position['avg_price'] - 1) * 100
            
            holdings.append({
                'symbol': symbol,
                'shares': position['shares'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'cost': cost,
                'market_value': market_value,
                'profit': profit,
                'profit_pct': profit_pct
            })
        
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'holdings_value': total_value - self.cash,
            'total_value': total_value,
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct,
            'holdings': holdings,
            'trade_count': len(self.trade_history)
        }
    
    def format_summary_message(self, current_prices: Dict[str, float] = None) -> str:
        """格式化帳戶摘要訊息（Telegram 格式）"""
        summary = self.get_summary(current_prices)
        
        # 計算總報酬顏色
        profit_emoji = "🔴" if summary['total_profit'] < 0 else "🟢" if summary['total_profit'] > 0 else "⚪"
        
        message = f"""
📊 **假設性交易帳戶摘要**

**初始資金：** {summary['initial_capital']:,.0f} TWD
**現金：** {summary['cash']:,.0f} TWD
**持股市值：** {summary['holdings_value']:,.0f} TWD
**總資產：** {summary['total_value']:,.0f} TWD

---

{profit_emoji} **總損益：** {summary['total_profit']:+,.0f} TWD ({summary['total_profit_pct']:+.2f}%)
**交易次數：** {summary['trade_count']} 次

---

**持倉明細：**
""".strip()
        
        if summary['holdings']:
            for holding in summary['holdings']:
                h_emoji = "🔴" if holding['profit'] < 0 else "🟢" if holding['profit'] > 0 else "⚪"
                message += f"""
{h_emoji} **{holding['symbol']}**
• 持股：{holding['shares']} 股
• 成本價：{holding['avg_price']:.2f} TWD
• 現價：{holding['current_price']:.2f} TWD
• 損益：{holding['profit']:+,.0f} TWD ({holding['profit_pct']:+.2f}%)
"""
        else:
            message += "\n（無持倉）"
        
        return message.strip()


def format_trade_message(trade: Dict) -> str:
    """格式化交易訊息（Telegram 格式）"""
    action_emoji = "🟢" if trade['action'] == 'buy' else "🔴"
    action_text = "買入" if trade['action'] == 'buy' else "賣出"
    
    message = f"""
{action_emoji} **交易執行：{action_text}**

**股票：** {trade['symbol']}
**價格：** {trade['price']:.2f} TWD
**股數：** {trade['shares']} 股
**時間：** {trade['timestamp'][:19]}

---
"""
    
    if trade['action'] == 'buy':
        message += f"""
**買入金額：** {trade['cost']:,.0f} TWD
**手續費：** {trade['commission']:,.0f} TWD
**總成本：** {trade['total_cost']:,.0f} TWD
**剩餘現金：** {trade['remaining_cash']:,.0f} TWD
""".strip()
    else:
        profit_emoji = "🔴" if trade['profit'] < 0 else "🟢" if trade['profit'] > 0 else "⚪"
        message += f"""
**賣出金額：** {trade['revenue']:,.0f} TWD
**手續費：** {trade['commission']:,.0f} TWD
**證交稅：** {trade['tax']:,.0f} TWD
**淨收入：** {trade['net_revenue']:,.0f} TWD
**剩餘現金：** {trade['remaining_cash']:,.0f} TWD

---

{profit_emoji} **本次損益：** {trade['profit']:+,.0f} TWD ({trade['profit_pct']:+.2f}%)
**成本價：** {trade['avg_cost']:.2f} TWD
""".strip()
    
    return message.strip()


# ==================== CLI ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='假設性交易測試')
    parser.add_argument('action', choices=['summary', 'reset'], help='動作')
    
    args = parser.parse_args()
    
    account = PaperTradingAccount()
    
    if args.action == 'summary':
        # 顯示帳戶摘要
        message = account.format_summary_message()
        print(message)
    
    elif args.action == 'reset':
        # 重置帳戶
        if account.log_file.exists():
            account.log_file.unlink()
        print("✅ 帳戶已重置")
