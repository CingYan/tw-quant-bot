#!/usr/bin/env python3
"""
回測引擎
"""

import logging
from typing import List, Dict, Type
from .strategy import Strategy
from .order import Portfolio
from .metrics import PerformanceMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """回測引擎"""
    
    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.001425,  # 台股手續費 0.1425%
        tax_rate: float = 0.003,            # 證交稅 0.3%
        slippage: float = 0.001             # 滑價 0.1%
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.slippage = slippage
        
        self.portfolio = Portfolio(initial_cash)
        self.strategy: Strategy = None
        self.equity_curve: List[Dict] = []
        self.metrics: PerformanceMetrics = None
    
    def add_strategy(self, strategy_class: Type[Strategy], **kwargs):
        """添加策略"""
        self.strategy = strategy_class(self.portfolio, **kwargs)
        logger.info(f"策略已載入: {strategy_class.__name__}")
    
    def load_data(self, symbol: str, data: List[Dict]):
        """載入股票資料"""
        if not self.strategy:
            raise ValueError("請先添加策略")
        
        self.strategy.load_data(symbol, data)
        logger.info(f"資料已載入: {symbol} ({len(data)} 筆)")
    
    def run(self):
        """執行回測"""
        if not self.strategy:
            raise ValueError("請先添加策略")
        
        if not self.strategy.data:
            raise ValueError("請先載入資料")
        
        logger.info("="*60)
        logger.info("開始回測...")
        logger.info(f"初始資金: {self.initial_cash:,.2f}")
        logger.info(f"手續費率: {self.commission_rate*100:.4f}%")
        logger.info(f"證交稅率: {self.tax_rate*100:.2f}%")
        logger.info(f"滑價: {self.slippage*100:.2f}%")
        logger.info("="*60)
        
        # 獲取最長的資料長度
        max_length = max(len(data) for data in self.strategy.data.values())
        
        # 回測開始回調
        self.strategy.on_start()
        
        # 逐 K 線回測
        for i in range(max_length):
            self.strategy.current_index = i
            
            # 執行策略邏輯
            self.strategy.on_bar()
            
            # 記錄權益曲線
            current_bar = None
            for symbol in self.strategy.data.keys():
                bar = self.strategy.get_current_bar(symbol)
                if bar:
                    current_bar = bar
                    break
            
            if current_bar:
                prices = {
                    symbol: self.strategy.get_current_bar(symbol)['close']
                    for symbol in self.strategy.data.keys()
                    if self.strategy.get_current_bar(symbol)
                }
                
                equity = self.portfolio.get_equity_curve(prices)
                
                self.equity_curve.append({
                    'date': current_bar['date'],
                    'equity': equity,
                    'cash': self.portfolio.cash,
                    'positions_value': equity - self.portfolio.cash
                })
        
        # 回測結束回調
        self.strategy.on_end()
        
        # 計算績效指標
        self.calculate_metrics()
        
        logger.info("="*60)
        logger.info("回測完成！")
        logger.info("="*60)
    
    def calculate_metrics(self):
        """計算績效指標"""
        self.metrics = PerformanceMetrics(
            equity_curve=self.equity_curve,
            initial_cash=self.initial_cash,
            orders=self.portfolio.orders,
            commission=self.portfolio.total_commission,
            tax=self.portfolio.total_tax
        )
    
    def get_summary(self) -> Dict:
        """獲取回測摘要"""
        if not self.metrics:
            return {}
        
        return self.metrics.get_summary()
    
    def print_summary(self):
        """打印回測摘要"""
        if not self.metrics:
            logger.warning("尚未執行回測")
            return
        
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("回測績效摘要")
        print("="*60)
        
        print(f"\n【基本資訊】")
        print(f"  初始資金: {summary['initial_cash']:,.2f}")
        print(f"  期末權益: {summary['final_equity']:,.2f}")
        print(f"  總報酬: {summary['total_return']:.2f}%")
        print(f"  總報酬金額: {summary['total_pnl']:,.2f}")
        
        print(f"\n【交易統計】")
        print(f"  總交易次數: {summary['total_trades']}")
        print(f"  獲利次數: {summary['winning_trades']}")
        print(f"  虧損次數: {summary['losing_trades']}")
        print(f"  勝率: {summary['win_rate']:.2f}%")
        
        print(f"\n【損益分析】")
        print(f"  平均獲利: {summary['avg_win']:,.2f}")
        print(f"  平均虧損: {summary['avg_loss']:,.2f}")
        print(f"  盈虧比: {summary['profit_factor']:.2f}")
        
        print(f"\n【風險指標】")
        print(f"  最大回撤: {summary['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        print(f"\n【成本分析】")
        print(f"  總手續費: {summary['total_commission']:,.2f}")
        print(f"  總證交稅: {summary['total_tax']:,.2f}")
        print(f"  總交易成本: {summary['total_cost']:,.2f}")
        
        print("\n" + "="*60)
    
    def print_signals(self, limit: int = 20):
        """打印交易訊號"""
        if not self.strategy or not self.strategy.signals:
            logger.warning("無交易訊號")
            return
        
        print("\n" + "="*60)
        print(f"交易訊號（最近 {limit} 筆）")
        print("="*60)
        
        signals = self.strategy.signals[-limit:]
        
        for signal in signals:
            print(f"{signal['date']} | {signal['symbol']} | "
                  f"{signal['type']:4s} | {signal['price']:>8.2f} | "
                  f"{signal['reason']}")
        
        print("="*60)
    
    def export_results(self, filepath: str):
        """匯出回測結果"""
        import json
        
        results = {
            'summary': self.get_summary(),
            'equity_curve': self.equity_curve,
            'signals': self.strategy.signals if self.strategy else [],
            'orders': [
                {
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'price': order.filled_price,
                    'commission': order.commission,
                    'tax': order.tax,
                    'status': order.status.value
                }
                for order in self.portfolio.orders
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"回測結果已匯出: {filepath}")
