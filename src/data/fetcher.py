#!/usr/bin/env python3
"""
台股資料擷取模組
使用 yfinance API 獲取台股即時與歷史資料
"""

import json
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import logging

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TWSEDataFetcher:
    """台股資料擷取器"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """初始化"""
        self.config = self._load_config(config_path)
        self.db_path = self.config["db"]["path"]
        self.cache_dir = Path(self.config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _load_config(self, config_path: str) -> dict:
        """載入配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _init_db(self):
        """初始化資料庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 建立股票資料表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        # 建立索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_date 
            ON stock_data(symbol, date)
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"資料庫初始化完成: {self.db_path}")
    
    def fetch_yahoo_finance(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Optional[List[Dict]]:
        """
        從 Yahoo Finance 獲取股票資料
        
        Args:
            symbol: 股票代碼（如 2330.TW）
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
        
        Returns:
            股票資料列表
        """
        try:
            # Yahoo Finance API (v7/v8)
            # 注意：這是簡化版本，實際使用建議用 yfinance 套件
            logger.info(f"獲取 {symbol} 資料: {start_date} ~ {end_date}")
            
            # 轉換日期為 Unix timestamp
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                f"?period1={start_ts}&period2={end_ts}&interval=1d"
                f"&events=history&includeAdjustedClose=true"
            )
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            # 解析資料
            if 'chart' not in data or 'result' not in data['chart']:
                logger.error(f"無效的回應格式: {symbol}")
                return None
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]
            adj_close = result['indicators'].get('adjclose', [{}])[0].get('adjclose', [None] * len(timestamps))
            
            # 組合資料
            stock_data = []
            for i, ts in enumerate(timestamps):
                date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                stock_data.append({
                    'symbol': symbol,
                    'date': date_str,
                    'open': quote['open'][i],
                    'high': quote['high'][i],
                    'low': quote['low'][i],
                    'close': quote['close'][i],
                    'volume': quote['volume'][i],
                    'adj_close': adj_close[i] if adj_close[i] else quote['close'][i]
                })
            
            logger.info(f"成功獲取 {len(stock_data)} 筆資料: {symbol}")
            return stock_data
            
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP 錯誤 {e.code}: {symbol}")
            return None
        except Exception as e:
            logger.error(f"獲取資料失敗 {symbol}: {str(e)}")
            return None
    
    def save_to_db(self, data: List[Dict]):
        """儲存資料到資料庫"""
        if not data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for row in data:
            cursor.execute('''
                INSERT OR REPLACE INTO stock_data 
                (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['symbol'],
                row['date'],
                row['open'],
                row['high'],
                row['low'],
                row['close'],
                row['volume'],
                row['adj_close']
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"成功儲存 {len(data)} 筆資料到資料庫")
    
    def fetch_and_save(
        self, 
        symbols: Optional[List[str]] = None,
        days: int = 365
    ):
        """獲取並儲存股票資料"""
        if symbols is None:
            symbols = self.config["data"]["symbols"]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        for symbol in symbols:
            logger.info(f"處理 {symbol}...")
            data = self.fetch_yahoo_finance(symbol, start_str, end_str)
            if data:
                self.save_to_db(data)
            else:
                logger.warning(f"跳過 {symbol}")
    
    def get_stock_data(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """從資料庫讀取股票資料"""
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


def main():
    """主程式"""
    # 初始化
    fetcher = TWSEDataFetcher()
    
    # 獲取最近一年的資料
    logger.info("開始獲取台股資料...")
    fetcher.fetch_and_save(days=365)
    
    # 測試讀取
    logger.info("\n測試讀取 2330.TW 最近 5 筆資料:")
    data = fetcher.get_stock_data("2330.TW")
    for row in data[-5:]:
        print(f"{row['date']}: Open={row['open']}, Close={row['close']}, Volume={row['volume']}")


if __name__ == "__main__":
    main()
