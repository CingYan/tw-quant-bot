#!/usr/bin/env python3
"""
基礎功能測試
測試資料擷取、清洗、技術指標計算
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.fetcher import TWSEDataFetcher
from data.processor import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fetcher():
    """測試資料擷取"""
    logger.info("=== 測試資料擷取 ===")
    
    fetcher = TWSEDataFetcher()
    
    # 測試單一股票
    symbol = "2330.TW"
    data = fetcher.fetch_yahoo_finance(symbol, "2025-01-01", "2025-01-31")
    
    if data:
        logger.info(f"✅ 成功獲取 {len(data)} 筆資料: {symbol}")
        logger.info(f"   範例資料: {data[0]}")
        
        # 儲存到資料庫
        fetcher.save_to_db(data)
        logger.info(f"✅ 資料已儲存到資料庫")
        
        return True
    else:
        logger.error(f"❌ 獲取資料失敗: {symbol}")
        return False


def test_processor():
    """測試資料處理"""
    logger.info("\n=== 測試資料處理 ===")
    
    processor = DataProcessor()
    symbol = "2330.TW"
    
    # 清洗資料
    count = processor.clean_data(symbol)
    logger.info(f"✅ 資料清洗完成，剩餘 {count} 筆")
    
    if count > 0:
        # 計算技術指標
        processor.calculate_technical_indicators(symbol)
        logger.info(f"✅ 技術指標計算完成")
        
        # 讀取處理後的資料
        data = processor.get_processed_data(symbol)
        
        if data:
            logger.info(f"✅ 成功讀取 {len(data)} 筆處理後資料")
            
            # 顯示最後一筆資料
            last = data[-1]
            logger.info(f"\n最新資料 ({last['date']}):")
            logger.info(f"  收盤價: {last.get('close')}")
            logger.info(f"  RSI: {last.get('rsi')}")
            logger.info(f"  MA5: {last.get('ma_5')}")
            logger.info(f"  MA20: {last.get('ma_20')}")
            logger.info(f"  成交量比率: {last.get('volume_ratio')}")
            
            return True
        else:
            logger.error(f"❌ 無法讀取處理後資料")
            return False
    else:
        logger.error(f"❌ 無資料可處理")
        return False


def main():
    """主測試流程"""
    logger.info("開始基礎功能測試...\n")
    
    results = []
    
    # 測試擷取
    results.append(("資料擷取", test_fetcher()))
    
    # 測試處理
    results.append(("資料處理", test_processor()))
    
    # 總結
    logger.info("\n" + "="*50)
    logger.info("測試結果總結")
    logger.info("="*50)
    
    for name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        logger.info(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        logger.info("\n🎉 所有測試通過！")
        logger.info("\n下一步：")
        logger.info("  1. 執行 `python3 main.py all` 更新所有股票資料")
        logger.info("  2. 開始實作階段二（Backtrader 回測引擎）")
    else:
        logger.info("\n⚠️  部分測試失敗，請檢查錯誤訊息")
    
    return all_passed


if __name__ == "__main__":
    # 創建必要目錄
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
