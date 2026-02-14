#!/usr/bin/env python3
"""
資料清洗與標準化模組
處理缺失值、異常值、特徵工程
"""

import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """資料處理器"""
    
    def __init__(self, db_path: str = "data/tw_quant.db"):
        self.db_path = db_path
    
    def clean_data(self, symbol: str) -> int:
        """
        清洗股票資料
        - 移除缺失值
        - 處理異常值（價格為0或負數）
        - 填補小範圍缺失（線性插值）
        
        Returns:
            清洗後的資料筆數
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. 移除價格為 NULL 或 0 的資料
        cursor.execute('''
            DELETE FROM stock_data
            WHERE symbol = ?
            AND (
                open IS NULL OR open <= 0 OR
                high IS NULL OR high <= 0 OR
                low IS NULL OR low <= 0 OR
                close IS NULL OR close <= 0 OR
                volume IS NULL OR volume < 0
            )
        ''', (symbol,))
        
        deleted = cursor.rowcount
        logger.info(f"{symbol}: 移除 {deleted} 筆異常資料")
        
        # 2. 檢查高低價邏輯
        cursor.execute('''
            DELETE FROM stock_data
            WHERE symbol = ?
            AND (high < low OR close > high OR close < low OR open > high OR open < low)
        ''', (symbol,))
        
        deleted_logic = cursor.rowcount
        if deleted_logic > 0:
            logger.warning(f"{symbol}: 移除 {deleted_logic} 筆邏輯錯誤資料")
        
        conn.commit()
        
        # 3. 計算清洗後的資料筆數
        cursor.execute('''
            SELECT COUNT(*) FROM stock_data WHERE symbol = ?
        ''', (symbol,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        logger.info(f"{symbol}: 清洗完成，剩餘 {count} 筆資料")
        return count
    
    def calculate_technical_indicators(self, symbol: str, period: int = 14):
        """
        計算技術指標
        - RSI (相對強弱指標)
        - MACD (指數平滑異同移動平均線)
        - MA (移動平均線)
        - Volume Ratio (成交量比率)
        
        注意：這是簡化版本，實際建議使用 TA-Lib 或 pandas-ta
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 先檢查表格是否有技術指標欄位
        cursor.execute("PRAGMA table_info(stock_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'rsi' not in columns:
            cursor.execute('''
                ALTER TABLE stock_data ADD COLUMN rsi REAL
            ''')
        
        if 'ma_5' not in columns:
            cursor.execute('''
                ALTER TABLE stock_data ADD COLUMN ma_5 REAL
            ''')
        
        if 'ma_20' not in columns:
            cursor.execute('''
                ALTER TABLE stock_data ADD COLUMN ma_20 REAL
            ''')
        
        if 'volume_ratio' not in columns:
            cursor.execute('''
                ALTER TABLE stock_data ADD COLUMN volume_ratio REAL
            ''')
        
        conn.commit()
        
        # 獲取歷史資料
        cursor.execute('''
            SELECT id, close, volume, date
            FROM stock_data
            WHERE symbol = ?
            ORDER BY date ASC
        ''', (symbol,))
        
        rows = cursor.fetchall()
        
        if len(rows) < period + 1:
            logger.warning(f"{symbol}: 資料不足，無法計算技術指標")
            conn.close()
            return
        
        # 計算 RSI (簡化版本)
        for i in range(period, len(rows)):
            closes = [row[1] for row in rows[i-period:i+1]]
            
            # 計算漲跌
            gains = []
            losses = []
            for j in range(1, len(closes)):
                change = closes[j] - closes[j-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            cursor.execute('''
                UPDATE stock_data
                SET rsi = ?
                WHERE id = ?
            ''', (rsi, rows[i][0]))
        
        # 計算移動平均線 (MA5, MA20)
        for i in range(len(rows)):
            row_id = rows[i][0]
            
            # MA5
            if i >= 4:
                ma5 = sum(row[1] for row in rows[i-4:i+1]) / 5
                cursor.execute('UPDATE stock_data SET ma_5 = ? WHERE id = ?', (ma5, row_id))
            
            # MA20
            if i >= 19:
                ma20 = sum(row[1] for row in rows[i-19:i+1]) / 20
                cursor.execute('UPDATE stock_data SET ma_20 = ? WHERE id = ?', (ma20, row_id))
        
        # 計算成交量比率（當日成交量 / 20日平均成交量）
        for i in range(19, len(rows)):
            row_id = rows[i][0]
            current_volume = rows[i][2]
            avg_volume = sum(row[2] for row in rows[i-19:i+1]) / 20
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                cursor.execute('''
                    UPDATE stock_data
                    SET volume_ratio = ?
                    WHERE id = ?
                ''', (volume_ratio, row_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"{symbol}: 技術指標計算完成")
    
    def get_processed_data(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """獲取處理後的資料（含技術指標）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM stock_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def export_to_json(self, symbol: str, output_path: str):
        """匯出處理後的資料為 JSON"""
        data = self.get_processed_data(symbol)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{symbol}: 匯出 {len(data)} 筆資料到 {output_path}")


def main():
    """主程式"""
    processor = DataProcessor()
    
    # 測試處理 2330.TW
    symbol = "2330.TW"
    
    logger.info(f"開始處理 {symbol}...")
    
    # 1. 清洗資料
    count = processor.clean_data(symbol)
    
    # 2. 計算技術指標
    processor.calculate_technical_indicators(symbol)
    
    # 3. 匯出處理後的資料
    processor.export_to_json(symbol, f"data/processed/{symbol}.json")
    
    # 4. 顯示最近 5 筆資料
    data = processor.get_processed_data(symbol)
    logger.info(f"\n最近 5 筆資料:")
    for row in data[-5:]:
        print(f"{row['date']}: Close={row.get('close')}, RSI={row.get('rsi', 'N/A')}, MA5={row.get('ma_5', 'N/A')}")


if __name__ == "__main__":
    main()
