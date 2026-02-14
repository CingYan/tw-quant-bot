#!/usr/bin/env python3
"""
回測引擎測試
測試 MA 交叉策略和 RSI 策略
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.processor import DataProcessor
from backtest import BacktestEngine, MAStrategy, RSIStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ma_strategy():
    """測試移動平均線策略"""
    logger.info("\n" + "="*60)
    logger.info("測試 MA 交叉策略")
    logger.info("="*60)
    
    # 1. 載入處理後的資料
    processor = DataProcessor()
    symbol = "2330.TW"
    
    data = processor.get_processed_data(symbol)
    
    if not data or len(data) < 30:
        logger.error(f"資料不足（需要至少 30 筆）: {len(data) if data else 0}")
        logger.info("請先執行: python3 main.py update --days 180")
        return False
    
    logger.info(f"資料已載入: {symbol} ({len(data)} 筆)")
    
    # 2. 創建回測引擎
    engine = BacktestEngine(
        initial_cash=1000000,
        commission_rate=0.001425,
        tax_rate=0.003
    )
    
    # 3. 添加策略（MA5 x MA20）
    engine.add_strategy(
        MAStrategy,
        short_period=5,
        long_period=20,
        position_pct=0.3  # 每次投入 30% 資金
    )
    
    # 4. 載入資料
    engine.load_data(symbol, data)
    
    # 5. 執行回測
    engine.run()
    
    # 6. 顯示結果
    engine.print_summary()
    engine.print_signals(limit=10)
    
    # 7. 匯出結果
    engine.export_results("data/backtest_ma_strategy.json")
    
    # 8. 驗證
    summary = engine.get_summary()
    
    if summary['total_trades'] > 0:
        logger.info("✅ MA 策略測試通過")
        return True
    else:
        logger.warning("⚠️  MA 策略未產生交易訊號")
        return False


def test_rsi_strategy():
    """測試 RSI 策略"""
    logger.info("\n" + "="*60)
    logger.info("測試 RSI 策略")
    logger.info("="*60)
    
    # 1. 載入處理後的資料
    processor = DataProcessor()
    symbol = "2330.TW"
    
    data = processor.get_processed_data(symbol)
    
    if not data or len(data) < 30:
        logger.error(f"資料不足: {len(data) if data else 0}")
        return False
    
    # 過濾掉 RSI 為 None 的資料
    data_with_rsi = [bar for bar in data if bar.get('rsi') is not None]
    
    if not data_with_rsi:
        logger.error("無 RSI 資料，請先執行資料處理")
        logger.info("執行: python3 main.py process")
        return False
    
    logger.info(f"資料已載入: {symbol} ({len(data_with_rsi)} 筆含 RSI)")
    
    # 2. 創建回測引擎
    engine = BacktestEngine(
        initial_cash=1000000,
        commission_rate=0.001425,
        tax_rate=0.003
    )
    
    # 3. 添加策略（RSI 超買超賣）
    engine.add_strategy(
        RSIStrategy,
        period=14,
        oversold=30,
        overbought=70,
        position_pct=0.3
    )
    
    # 4. 載入資料
    engine.load_data(symbol, data_with_rsi)
    
    # 5. 執行回測
    engine.run()
    
    # 6. 顯示結果
    engine.print_summary()
    engine.print_signals(limit=10)
    
    # 7. 匯出結果
    engine.export_results("data/backtest_rsi_strategy.json")
    
    # 8. 驗證
    summary = engine.get_summary()
    
    if summary['total_trades'] > 0:
        logger.info("✅ RSI 策略測試通過")
        return True
    else:
        logger.warning("⚠️  RSI 策略未產生交易訊號")
        return False


def compare_strategies():
    """比較兩種策略"""
    logger.info("\n" + "="*60)
    logger.info("策略比較")
    logger.info("="*60)
    
    import json
    
    try:
        with open("data/backtest_ma_strategy.json", 'r') as f:
            ma_results = json.load(f)
        
        with open("data/backtest_rsi_strategy.json", 'r') as f:
            rsi_results = json.load(f)
        
        ma_summary = ma_results['summary']
        rsi_summary = rsi_results['summary']
        
        print(f"\n{'指標':<20} {'MA 策略':>15} {'RSI 策略':>15}")
        print("-" * 52)
        print(f"{'總報酬率':<20} {ma_summary['total_return']:>14.2f}% {rsi_summary['total_return']:>14.2f}%")
        print(f"{'最大回撤':<20} {ma_summary['max_drawdown']:>14.2f}% {rsi_summary['max_drawdown']:>14.2f}%")
        print(f"{'Sharpe Ratio':<20} {ma_summary['sharpe_ratio']:>15.2f} {rsi_summary['sharpe_ratio']:>15.2f}")
        print(f"{'勝率':<20} {ma_summary['win_rate']:>14.2f}% {rsi_summary['win_rate']:>14.2f}%")
        print(f"{'交易次數':<20} {ma_summary['total_trades']:>15} {rsi_summary['total_trades']:>15}")
        print(f"{'交易成本':<20} {ma_summary['total_cost']:>15,.2f} {rsi_summary['total_cost']:>15,.2f}")
        
        print("\n" + "="*60)
        
        # 判斷哪個策略更好
        if ma_summary['total_return'] > rsi_summary['total_return']:
            logger.info("🏆 MA 策略表現較佳")
        else:
            logger.info("🏆 RSI 策略表現較佳")
        
    except FileNotFoundError:
        logger.error("請先執行策略測試")


def main():
    """主測試流程"""
    logger.info("開始回測引擎測試...\n")
    
    results = []
    
    # 測試 MA 策略
    results.append(("MA 策略", test_ma_strategy()))
    
    # 測試 RSI 策略
    results.append(("RSI 策略", test_rsi_strategy()))
    
    # 比較策略
    if all(result for _, result in results):
        compare_strategies()
    
    # 總結
    logger.info("\n" + "="*60)
    logger.info("測試結果總結")
    logger.info("="*60)
    
    for name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n🎉 所有測試通過！")
        logger.info("\n下一步：")
        logger.info("  1. 查看回測結果：data/backtest_*.json")
        logger.info("  2. 開始實作階段三（智能層）")
    else:
        logger.info("\n⚠️  部分測試失敗")
        logger.info("提示：請先執行 python3 main.py all 更新資料")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
