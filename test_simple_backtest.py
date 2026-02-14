#!/usr/bin/env python3
"""
簡單回測測試 - 買入持有策略
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.processor import DataProcessor
from backtest import BacktestEngine, Strategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuyAndHoldStrategy(Strategy):
    """買入持有策略（測試用）"""
    
    def __init__(self, portfolio, position_pct=0.5):
        super().__init__(portfolio)
        self.position_pct = position_pct
        self.bought = False
    
    def on_bar(self):
        """在第一天買入，持有到最後"""
        for symbol in self.data.keys():
            bar = self.get_current_bar(symbol)
            
            if not bar:
                continue
            
            position = self.portfolio.get_position(symbol)
            
            # 第一次執行時買入
            if not self.bought and position.quantity == 0:
                quantity = self.calculate_position_size(symbol, self.position_pct)
                
                if quantity > 0:
                    order = self.buy(symbol, quantity)
                    self.portfolio.execute_order(order)
                    
                    self.record_signal(
                        symbol,
                        'BUY',
                        f'買入持有（第一天）',
                        {'price': bar['close']}
                    )
                    
                    self.bought = True
                    logger.info(f"買入 {symbol}: {quantity} 股 @ {bar['close']:.2f}")
    
    def on_end(self):
        """回測結束時賣出（可選）"""
        for symbol in self.data.keys():
            position = self.portfolio.get_position(symbol)
            
            if position.quantity > 0:
                order = self.close_position(symbol)
                
                if order:
                    self.portfolio.execute_order(order)
                    
                    bar = self.get_current_bar(symbol)
                    self.record_signal(
                        symbol,
                        'SELL',
                        f'賣出（最後一天）',
                        {'price': bar['close']}
                    )
                    
                    logger.info(f"賣出 {symbol}: {position.quantity} 股 @ {bar['close']:.2f}")


def main():
    """主測試"""
    logger.info("="*60)
    logger.info("簡單回測測試 - 買入持有策略")
    logger.info("="*60)
    
    # 1. 載入資料
    processor = DataProcessor()
    symbol = "2330.TW"
    
    data = processor.get_processed_data(symbol)
    
    if not data or len(data) < 10:
        logger.error(f"資料不足: {len(data) if data else 0}")
        return False
    
    logger.info(f"資料已載入: {symbol} ({len(data)} 筆)")
    logger.info(f"期間: {data[0]['date']} ~ {data[-1]['date']}")
    logger.info(f"起始價: {data[0]['close']:.2f}, 結束價: {data[-1]['close']:.2f}")
    
    # 2. 創建回測引擎
    engine = BacktestEngine(
        initial_cash=1000000,
        commission_rate=0.001425,
        tax_rate=0.003
    )
    
    # 3. 添加策略
    engine.add_strategy(BuyAndHoldStrategy, position_pct=0.5)
    
    # 4. 載入資料
    engine.load_data(symbol, data)
    
    # 5. 執行回測
    engine.run()
    
    # 6. 顯示結果
    engine.print_summary()
    engine.print_signals()
    
    # 7. 匯出結果
    engine.export_results("data/backtest_buy_hold.json")
    
    # 8. 驗證
    summary = engine.get_summary()
    
    if summary['total_trades'] > 0:
        logger.info("✅ 回測引擎測試通過")
        return True
    else:
        logger.error("❌ 未產生交易")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
