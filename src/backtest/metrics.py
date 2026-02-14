#!/usr/bin/env python3
"""
績效評估模組
"""

import math
from typing import List, Dict
from .order import Order, OrderSide


class PerformanceMetrics:
    """績效指標計算器"""
    
    def __init__(
        self,
        equity_curve: List[Dict],
        initial_cash: float,
        orders: List[Order],
        commission: float = 0.0,
        tax: float = 0.0
    ):
        self.equity_curve = equity_curve
        self.initial_cash = initial_cash
        self.orders = orders
        self.total_commission = commission
        self.total_tax = tax
    
    @property
    def final_equity(self) -> float:
        """期末權益"""
        if not self.equity_curve:
            return self.initial_cash
        return self.equity_curve[-1]['equity']
    
    @property
    def total_return(self) -> float:
        """總報酬率 (%)"""
        return ((self.final_equity - self.initial_cash) / self.initial_cash) * 100
    
    @property
    def total_pnl(self) -> float:
        """總損益"""
        return self.final_equity - self.initial_cash
    
    @property
    def max_drawdown(self) -> float:
        """最大回撤 (%)"""
        if not self.equity_curve:
            return 0.0
        
        peak = self.initial_cash
        max_dd = 0.0
        
        for point in self.equity_curve:
            equity = point['equity']
            
            if equity > peak:
                peak = equity
            
            drawdown = ((peak - equity) / peak) * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        """Sharpe Ratio（假設無風險利率為 0）"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # 計算日報酬率
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i-1]['equity']
            curr = self.equity_curve[i]['equity']
            ret = (curr - prev) / prev
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        # 計算平均報酬和標準差
        avg_return = sum(returns) / len(returns)
        
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        # 年化 Sharpe Ratio（假設 252 交易日）
        sharpe = (avg_return / std_dev) * math.sqrt(252)
        
        return sharpe
    
    @property
    def total_trades(self) -> int:
        """總交易次數（成對計算：買入+賣出=1次交易）"""
        buy_count = sum(1 for order in self.orders if order.side == OrderSide.BUY)
        sell_count = sum(1 for order in self.orders if order.side == OrderSide.SELL)
        return min(buy_count, sell_count)
    
    @property
    def winning_trades(self) -> int:
        """獲利交易次數"""
        # 配對買賣訂單計算損益
        trades = self._pair_trades()
        return sum(1 for trade in trades if trade['pnl'] > 0)
    
    @property
    def losing_trades(self) -> int:
        """虧損交易次數"""
        trades = self._pair_trades()
        return sum(1 for trade in trades if trade['pnl'] < 0)
    
    @property
    def win_rate(self) -> float:
        """勝率 (%)"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def avg_win(self) -> float:
        """平均獲利"""
        trades = self._pair_trades()
        winning = [t['pnl'] for t in trades if t['pnl'] > 0]
        
        if not winning:
            return 0.0
        
        return sum(winning) / len(winning)
    
    @property
    def avg_loss(self) -> float:
        """平均虧損"""
        trades = self._pair_trades()
        losing = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        if not losing:
            return 0.0
        
        return sum(losing) / len(losing)
    
    @property
    def profit_factor(self) -> float:
        """盈虧比（總獲利 / 總虧損）"""
        trades = self._pair_trades()
        
        total_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        if total_loss == 0:
            return float('inf') if total_win > 0 else 0.0
        
        return total_win / total_loss
    
    def _pair_trades(self) -> List[Dict]:
        """配對買賣訂單計算損益"""
        trades = []
        positions = {}  # symbol -> [buy_orders]
        
        for order in self.orders:
            symbol = order.symbol
            
            if symbol not in positions:
                positions[symbol] = []
            
            if order.side == OrderSide.BUY:
                positions[symbol].append(order)
            
            else:  # SELL
                # 配對最早的買入訂單（FIFO）
                if positions[symbol]:
                    buy_order = positions[symbol].pop(0)
                    
                    # 計算損益（簡化：假設買賣數量相同）
                    buy_cost = buy_order.total_cost
                    sell_revenue = (
                        order.filled_price * order.quantity 
                        - order.commission 
                        - order.tax
                    )
                    
                    pnl = sell_revenue - buy_cost
                    
                    trades.append({
                        'symbol': symbol,
                        'buy_price': buy_order.filled_price,
                        'sell_price': order.filled_price,
                        'quantity': order.quantity,
                        'pnl': pnl
                    })
        
        return trades
    
    def get_summary(self) -> Dict:
        """獲取績效摘要"""
        return {
            'initial_cash': self.initial_cash,
            'final_equity': self.final_equity,
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'total_commission': self.total_commission,
            'total_tax': self.total_tax,
            'total_cost': self.total_commission + self.total_tax
        }
