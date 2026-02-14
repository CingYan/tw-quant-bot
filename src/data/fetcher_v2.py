#!/usr/bin/env python3
"""
台股資料擷取模組 v2 - 使用 yfinance
改進版本，使用 yfinance 套件替代手動 API 調用
"""

import json
import sqlite3
import yfinance as yf
import pandas as pd
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


class TWSEDataFetcherV2:
    """台股資料擷取器 v2（使用 yfinance）"""
    
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
    
    def fetch_yfinance(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        使用 yfinance 獲取股票資料
        
        Args:
            symbol: 股票代碼（如 2330.TW）
            start_date: 開始日期 (YYYY-MM-DD)，可選
            end_date: 結束日期 (YYYY-MM-DD)，可選
            period: 時間範圍（如 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
        
        Returns:
            DataFrame 或 None
        """
        try:
            logger.info(f"使用 yfinance 獲取 {symbol} 資料...")
            
            ticker = yf.Ticker(symbol)
            
            # 使用日期範圍或期間
            if start_date and end_date:
                hist = ticker.history(start=start_date, end=end_date)
            else:
                hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"{symbol}: 無資料")
                return None
            
            # 重置索引（將日期從索引變為欄位）
            hist.reset_index(inplace=True)
            
            # 重命名欄位（統一小寫）
            hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # 添加調整後收盤價（如果沒有的話）
            if 'adj_close' not in hist.columns:
                hist['adj_close'] = hist['close']
            
            # 格式化日期
            hist['date'] = pd.to_datetime(hist['date']).dt.strftime('%Y-%m-%d')
            
            # 添加股票代碼
            hist['symbol'] = symbol
            
            logger.info(f"成功獲取 {len(hist)} 筆資料: {symbol}")
            
            return hist
            
        except Exception as e:
            logger.error(f"獲取資料失敗 {symbol}: {str(e)}")
            return None
    
    def save_to_db(self, df: pd.DataFrame):
        """儲存 DataFrame 到資料庫"""
        if df is None or df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 只選擇需要的欄位
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        df_filtered = df[columns]
        
        # 使用 INSERT OR REPLACE 處理重複資料
        for _, row in df_filtered.iterrows():
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
        
        logger.info(f"成功儲存 {len(df_filtered)} 筆資料到資料庫")
    
    def fetch_and_save(
        self, 
        symbols: Optional[List[str]] = None,
        period: str = "1y"
    ):
        """獲取並儲存股票資料"""
        if symbols is None:
            symbols = self.config["data"]["symbols"]
        
        for symbol in symbols:
            logger.info(f"處理 {symbol}...")
            
            df = self.fetch_yfinance(symbol, period=period)
            
            if df is not None:
                # 先刪除舊資料（避免重複）
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM stock_data WHERE symbol = ?', (symbol,))
                conn.commit()
                conn.close()
                
                # 儲存新資料
                self.save_to_db(df)
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
    
    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """獲取最新價格（即時資料）"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'name': info.get('longName') or info.get('shortName')
            }
        except Exception as e:
            logger.error(f"獲取即時價格失敗 {symbol}: {str(e)}")
            return None


def main():
    """主程式"""
    # 初始化
    fetcher = TWSEDataFetcherV2()
    
    # 測試獲取台積電資料
    logger.info("測試 yfinance 資料擷取...")
    
    df = fetcher.fetch_yfinance("2330.TW", period="1y")
    
    if df is not None:
        print(f"\n成功獲取 {len(df)} 筆資料")
        print(df.tail(10))
        
        # 儲存到資料庫
        fetcher.save_to_db(df)
        
        # 測試讀取
        data = fetcher.get_stock_data("2330.TW")
        print(f"\n從資料庫讀取 {len(data)} 筆資料")
        
        # 測試即時價格
        latest = fetcher.get_latest_price("2330.TW")
        if latest:
            print(f"\n最新價格: {latest['name']} - {latest['price']} TWD")


if __name__ == "__main__":
    main()
