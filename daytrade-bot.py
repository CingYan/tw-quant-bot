#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股當沖模擬交易 Bot - v2.0.0
22+個技術指標 + 評分系統 + 動態ATR停損停利
新增: Shioaji即時報價 + FinMind法人買賣超 + 期交所Put/Call Ratio
"""

__version__ = "2.1.0"

import sys
import os
import logging
import time
import json
import subprocess
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
import requests
from bs4 import BeautifulSoup

# ============================================================================
# 依賴檢查和安裝
# ============================================================================
def check_and_install_dependencies():
    """檢查並安裝所需的依賴"""
    required_packages = {
        'yfinance': 'yfinance',
        'pandas': 'pandas',
        'ta': 'ta',
        'numpy': 'numpy',
        'requests': 'requests'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[INFO] 正在安裝缺失的套件: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("[INFO] 依賴安裝完成")
        except Exception as e:
            print(f"[ERROR] 無法安裝依賴: {e}")
            print(f"請手動執行: pip install {' '.join(missing)}")
            sys.exit(1)

check_and_install_dependencies()

import pandas as pd
import numpy as np
import yfinance as yf
import ta

# ============================================================================
# Credentials 載入
# ============================================================================
def load_credentials() -> Dict:
    """從 credentials 檔案讀取 API Keys"""
    cred_path = Path('/home/node/clawd/credentials/8df795c8dca1.json')
    if cred_path.exists():
        try:
            with open(cred_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] 讀取 credentials 失敗: {e}")
            return {}
    return {}

CREDENTIALS = load_credentials()

# ============================================================================
# 配置和常量
# ============================================================================
CONFIG = {
    'MEMORY_DIR': Path('/home/node/clawd/memory'),
    'PICKS_FILE': Path('/home/node/clawd/memory/daytrade-picks.md'),
    'POSITIONS_FILE': Path('/home/node/clawd/memory/daytrade-positions.md'),
    'RESULTS_FILE': Path('/home/node/clawd/memory/daytrade-results.md'),
    'ALERT_FILE': Path('/home/node/clawd/memory/daytrade-alert.md'),
    'LOG_FILE': Path('/home/node/clawd/scripts/daytrade-bot.log'),
    
    # 交易參數
    'START_HOUR': 9,      # 9:00 開市
    'END_HOUR': 13,
    'END_MINUTE': 30,
    'NO_OPEN_AFTER_HOUR': 13,
    'NO_OPEN_AFTER_MINUTE': 0,
    'FORCE_CLOSE_HOUR': 13,
    'FORCE_CLOSE_MINUTE': 25,
    
    # 交易設定（現在用ATR替代固定百分比）
    'SCAN_INTERVAL': 0.5,      # 無持倉時：每500ms掃描一次（找買入機會）
    'MONITOR_INTERVAL': 0.2,   # 有持倉時：每200ms監控一次（停損停利）
    'VOLUME_MULTIPLIER': 1.5,  # 成交量倍數
    'BUY_SCORE_THRESHOLD': 60, # 買入評分門檻 (滿分100)
}

# 評分系統權重配置（已升級到 22+ 指標）
SIGNAL_WEIGHTS = {
    'ma_cross': 10,           # 均線交叉 (5MA > 10MA)
    'rsi': 8,                 # RSI 超賣/強勢
    'volume': 10,             # 成交量爆發
    'vwap': 8,                # VWAP
    'macd': 10,               # MACD 交叉
    'bollinger': 7,           # 布林帶
    'candle_pattern': 5,      # K棒型態
    'stochastic': 6,          # Stochastic KD
    'adx': 7,                 # ADX 趨勢強度
    'cci': 5,                 # CCI
    'williams': 5,            # Williams %R
    'obv': 6,                 # OBV
    'mfi': 5,                 # MFI
    'psar': 6,                # Parabolic SAR
    'ichimoku': 8,            # Ichimoku
    'donchian': 5,            # Donchian Channel
    'ema_multi': 8,           # EMA 多重確認
    'news_sentiment': 10,     # Marketaux NLP 新聞情緒分析
    # === 新增 4 個指標 (v2.0.0) ===
    'order_flow': 8,          # Shioaji 掛單流量 (買方掛單 > 賣方掛單)
    'delta': 7,               # Shioaji Tick Delta (累積 delta 為正)
    'institutional': 8,       # FinMind 三大法人買賣超
    'put_call_ratio': 7,      # 期交所 Put/Call Ratio
}

# 初始化日誌
def setup_logging():
    """設置日誌系統"""
    logger = logging.getLogger('daytrade_bot')
    logger.setLevel(logging.DEBUG)
    
    # 確保日誌目錄存在
    CONFIG['LOG_FILE'].parent.mkdir(parents=True, exist_ok=True)
    
    # 日誌格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 檔案處理器
    fh = logging.FileHandler(CONFIG['LOG_FILE'], encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台處理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ============================================================================
# 外部 API 配置和快取
# ============================================================================
MARKETAUX_API_KEY = CREDENTIALS.get('marketaux', {}).get('api_key', '')
MARKETAUX_API_URL = "https://api.marketaux.com/v1/news/all"

# Shioaji 配置
SHIOAJI_API_KEY = CREDENTIALS.get('shioaji', {}).get('api_key', '')
SHIOAJI_SECRET_KEY = CREDENTIALS.get('shioaji', {}).get('secret_key', '')

# FinMind 配置
FINMIND_API_TOKEN = CREDENTIALS.get('finmind', {}).get('api_token', '')
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
NEWS_SENTIMENT_CACHE = {}  # 快取: {symbol: (timestamp, sentiment_score)}
NEWS_CACHE_EXPIRY = 3600   # 快取有效期：1小時（秒）

# Shioaji 快取
SHIOAJI_ORDER_FLOW_CACHE = {}  # {symbol: (timestamp, order_flow_score, delta)}
SHIOAJI_CACHE_EXPIRY = 300  # 5分鐘

# FinMind 快取
FINMIND_INSTITUTIONAL_CACHE = {}  # {symbol: (timestamp, institutional_score)}
FINMIND_CACHE_EXPIRY = 86400  # 24小時

# 期交所 Put/Call Ratio 快取
TAIFEX_PC_RATIO_CACHE = {}  # {date: (timestamp, pc_ratio, signal_score)}
TAIFEX_CACHE_EXPIRY = 86400  # 24小時

# ============================================================================
# Shioaji 即時報價 + Order Flow
# ============================================================================
def get_shioaji_order_flow(stock_symbol: str) -> Tuple[float, float]:
    """
    從 Shioaji 獲取掛單流量指標（Order Flow + Delta）
    
    Returns:
        (order_flow_score, delta_score) - 均為 0 ~ 10 的評分
    """
    try:
        if not SHIOAJI_API_KEY or not SHIOAJI_SECRET_KEY:
            logger.debug("Shioaji API keys 未配置，跳過")
            return 0.0, 0.0
        
        # 檢查快取
        current_time = time.time()
        if stock_symbol in SHIOAJI_ORDER_FLOW_CACHE:
            cache_timestamp, cached_order_flow, cached_delta = SHIOAJI_ORDER_FLOW_CACHE[stock_symbol]
            if current_time - cache_timestamp < SHIOAJI_CACHE_EXPIRY:
                logger.debug(f"{stock_symbol} 使用快取 Shioaji: order_flow={cached_order_flow:.1f}, delta={cached_delta:.1f}")
                return cached_order_flow, cached_delta
        
        # 嘗試匯入 Shioaji
        try:
            import shioaji
        except ImportError:
            logger.warning("Shioaji 模組未安裝，跳過 order flow 計算")
            return 0.0, 0.0
        
        # 登入 Shioaji
        api = shioaji.Shioaji()
        api.login(
            api_key=SHIOAJI_API_KEY,
            secret_key=SHIOAJI_SECRET_KEY,
            contracts_cb=lambda x: None
        )
        
        # 提取股票代碼（去掉 .TW）
        ticker = stock_symbol.split('.')[0]
        
        # 取得即時五檔報價
        quote = api.quote.quote(api.Contracts.Stocks[ticker])
        
        if not quote:
            logger.warning(f"{stock_symbol} 無法取得即時報價")
            return 0.0, 0.0
        
        # 提取買賣掛單量
        bid_volumes = quote.get('Bid', [])
        ask_volumes = quote.get('Ask', [])
        
        total_bid = sum([v[1] for v in bid_volumes]) if bid_volumes else 0
        total_ask = sum([v[1] for v in ask_volumes]) if ask_volumes else 0
        
        # Order Flow 評分：買方掛單 > 賣方掛單 = 偏多
        if total_bid + total_ask > 0:
            bid_ratio = total_bid / (total_bid + total_ask)
            order_flow_score = min(10, (bid_ratio - 0.5) * 20) if bid_ratio > 0.5 else 0  # 買方掛單 > 50% = 加分
        else:
            order_flow_score = 0.0
        
        # Delta 評分：累積 tick delta 為正（買壓大於賣壓）
        # 簡化版：以最新成交價相對於中點的位置判斷
        bid_price = bid_volumes[0][0] if bid_volumes else 0
        ask_price = ask_volumes[0][0] if ask_volumes else 0
        last_price = quote.get('LastPrice', 0)
        
        if bid_price > 0 and ask_price > 0:
            midpoint = (bid_price + ask_price) / 2
            if last_price > midpoint:
                delta_score = min(10, (last_price - midpoint) / (ask_price - bid_price) * 20)  # 成交在買方 = delta 正
            else:
                delta_score = 0.0
        else:
            delta_score = 0.0
        
        # 更新快取
        SHIOAJI_ORDER_FLOW_CACHE[stock_symbol] = (current_time, order_flow_score, delta_score)
        
        logger.info(f"{stock_symbol} Shioaji 指標: order_flow={order_flow_score:.1f}, delta={delta_score:.1f}")
        
        api.logout()
        return order_flow_score, delta_score
    
    except Exception as e:
        logger.warning(f"{stock_symbol} 取得 Shioaji 指標失敗: {e}")
        return 0.0, 0.0

# ============================================================================
# FinMind 歷史數據 + 三大法人買賣超
# ============================================================================
def get_finmind_institutional(stock_symbol: str) -> float:
    """
    從 FinMind 獲取三大法人買賣超訊息
    
    Returns:
        institutional_score - 0 ~ 8 的評分（連續買超 = 高分）
    """
    try:
        if not FINMIND_API_TOKEN:
            logger.debug("FinMind API token 未配置，跳過")
            return 0.0
        
        # 檢查快取
        current_time = time.time()
        if stock_symbol in FINMIND_INSTITUTIONAL_CACHE:
            cache_timestamp, cached_score = FINMIND_INSTITUTIONAL_CACHE[stock_symbol]
            if current_time - cache_timestamp < FINMIND_CACHE_EXPIRY:
                logger.debug(f"{stock_symbol} 使用快取 FinMind: score={cached_score:.1f}")
                return cached_score
        
        # 提取股票代碼（去掉 .TW）
        ticker = stock_symbol.split('.')[0]
        
        # 查詢三大法人買賣超
        params = {
            'dataset': 'TaiwanStockInstitutionalBuySell',
            'data_id': ticker,
            'start_date': (datetime.now() - __import__('datetime').timedelta(days=5)).strftime('%Y-%m-%d'),
            'token': FINMIND_API_TOKEN
        }
        
        logger.debug(f"查詢 {stock_symbol} 的三大法人買賣超...")
        response = requests.get(FINMIND_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # 檢查 API 回應
        if data.get('status') != 'ok' or 'data' not in data:
            logger.warning(f"{stock_symbol} FinMind API 返回異常: {data.get('msg', 'Unknown')}")
            return 0.0
        
        records = data.get('data', [])
        if not records:
            logger.warning(f"{stock_symbol} 無三大法人數據")
            return 0.0
        
        # 檢查最近數天是否連續買超（外資 + 投信）
        institutional_score = 0.0
        consecutive_buy_days = 0
        
        for record in records[-3:]:  # 檢查最近 3 天
            foreign_buy = float(record.get('ForeignBuySell', 0))
            investment_buy = float(record.get('InvestmentBuySell', 0))
            
            if foreign_buy > 0 and investment_buy > 0:
                consecutive_buy_days += 1
        
        # 連續買超天數 = 評分
        institutional_score = min(8, consecutive_buy_days * 2.5)  # 最高 8 分
        
        # 更新快取
        FINMIND_INSTITUTIONAL_CACHE[stock_symbol] = (current_time, institutional_score)
        
        logger.info(f"{stock_symbol} FinMind 三大法人: 連續買超 {consecutive_buy_days} 天 (評分: {institutional_score:.1f})")
        
        return institutional_score
    
    except requests.exceptions.RequestException as e:
        logger.warning(f"{stock_symbol} 獲取 FinMind 數據失敗 (網路錯誤): {e}")
        return 0.0
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"{stock_symbol} 解析 FinMind 數據失敗: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"{stock_symbol} 三大法人分析異常: {e}")
        return 0.0

# ============================================================================
# 期交所 Put/Call Ratio 爬蟲
# ============================================================================
def get_taifex_put_call_ratio() -> float:
    """
    爬取期交所 Put/Call Ratio
    
    Returns:
        put_call_score - 0 ~ 7 的評分（偏多 = 高分）
    """
    try:
        # 檢查快取（每天只爬一次）
        current_time = time.time()
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today in TAIFEX_PC_RATIO_CACHE:
            cache_timestamp, cached_score = TAIFEX_PC_RATIO_CACHE[today]
            if current_time - cache_timestamp < TAIFEX_CACHE_EXPIRY:
                logger.debug(f"使用快取 TAIFEX P/C Ratio: score={cached_score:.1f}")
                return cached_score
        
        # 爬取期交所網頁
        url = "https://www.taifex.com.tw/cht/3/pcRatio"
        logger.debug(f"爬取期交所 Put/Call Ratio...")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 使用 BeautifulSoup 解析
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 尋找 P/C Ratio 數據（通常在表格中）
        # 期交所頁面結構可能變動，這裡是一個基本實現
        pc_ratio = None
        
        # 嘗試從表格中提取
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # 查找包含 "Put/Call" 或 "P/C" 的行
                    if 'put' in cells[0].get_text().lower() or 'call' in cells[0].get_text().lower():
                        try:
                            pc_ratio = float(cells[1].get_text().strip())
                            break
                        except:
                            pass
            if pc_ratio:
                break
        
        # 若無法從表格取得，嘗試從 text 中提取
        if not pc_ratio:
            import re
            matches = re.findall(r'[Pp]/[Cc]\s*[:=]\s*([\d.]+)', response.text)
            if matches:
                pc_ratio = float(matches[0])
        
        if not pc_ratio:
            logger.warning("無法解析期交所 P/C Ratio")
            return 0.0
        
        # 評分邏輯
        # P/C < 0.7: 市場偏多 (評分 7)
        # P/C 0.7 ~ 1.0: 中性偏多 (評分 5)
        # P/C 1.0 ~ 1.3: 中性 (評分 0)
        # P/C > 1.3 後回落: 恐慌反轉 (評分 7)
        
        put_call_score = 0.0
        
        if pc_ratio < 0.7:
            put_call_score = 7.0  # 偏多
        elif pc_ratio < 1.0:
            put_call_score = 5.0  # 中性偏多
        elif pc_ratio > 1.3:
            put_call_score = 7.0  # 恐慌反轉
        
        # 更新快取
        TAIFEX_PC_RATIO_CACHE[today] = (current_time, put_call_score)
        
        logger.info(f"期交所 P/C Ratio: {pc_ratio:.2f} (評分: {put_call_score:.1f})")
        
        return put_call_score
    
    except requests.exceptions.RequestException as e:
        logger.warning(f"爬取期交所 P/C Ratio 失敗 (網路錯誤): {e}")
        return 0.0
    except Exception as e:
        logger.error(f"期交所 P/C Ratio 分析異常: {e}")
        return 0.0

def get_news_sentiment(stock_symbol: str) -> float:
    """
    從 Marketaux API 獲取股票新聞情緒分析
    
    Args:
        stock_symbol: 股票代碼 (如 "2330.TW")
    
    Returns:
        情緒分數 (0.0 ~ 1.0)，失敗時返回 0.0
    """
    try:
        # 檢查快取
        current_time = time.time()
        if stock_symbol in NEWS_SENTIMENT_CACHE:
            cache_timestamp, cached_sentiment = NEWS_SENTIMENT_CACHE[stock_symbol]
            if current_time - cache_timestamp < NEWS_CACHE_EXPIRY:
                logger.debug(f"{stock_symbol} 使用快取新聞情緒: {cached_sentiment:.3f}")
                return cached_sentiment
        
        # 提取股票代碼（去掉 .TW）
        ticker = stock_symbol.split('.')[0]
        
        # 呼叫 Marketaux API
        params = {
            'symbols': ticker,
            'filter_entities': 'true',
            'language': 'en,zh',
            'api_token': MARKETAUX_API_KEY
        }
        
        logger.debug(f"查詢 {stock_symbol} 新聞情緒...")
        response = requests.get(MARKETAUX_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # 檢查 API 回應
        if 'data' not in data or not isinstance(data['data'], list):
            logger.warning(f"{stock_symbol} API 返回格式異常")
            return 0.0
        
        news_list = data['data']
        
        # 若無新聞，返回 0.0
        if not news_list:
            logger.warning(f"{stock_symbol} 無相關新聞")
            NEWS_SENTIMENT_CACHE[stock_symbol] = (current_time, 0.0)
            return 0.0
        
        # 取最近 5 篇新聞的平均情緒分數
        sentiments = []
        for news in news_list[:5]:
            if 'sentiment' in news:
                sentiment_value = news['sentiment']
                if isinstance(sentiment_value, (int, float)):
                    sentiments.append(sentiment_value)
        
        if not sentiments:
            logger.warning(f"{stock_symbol} 新聞無情緒分數")
            NEWS_SENTIMENT_CACHE[stock_symbol] = (current_time, 0.0)
            return 0.0
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        avg_sentiment = max(0.0, min(1.0, avg_sentiment))  # 限制在 0.0 ~ 1.0
        
        # 更新快取
        NEWS_SENTIMENT_CACHE[stock_symbol] = (current_time, avg_sentiment)
        
        logger.info(f"{stock_symbol} 新聞情緒分數: {avg_sentiment:.3f} (基於 {len(sentiments)} 篇新聞)")
        return avg_sentiment
    
    except requests.exceptions.RequestException as e:
        logger.warning(f"{stock_symbol} 獲取新聞失敗 (網路錯誤): {e}")
        return 0.0
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"{stock_symbol} 解析新聞數據失敗: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"{stock_symbol} 新聞情緒分析異常: {e}")
        return 0.0

# ============================================================================
# 檔案操作函數
# ============================================================================
def read_picks() -> List[str]:
    """讀取候選股清單"""
    try:
        if CONFIG['PICKS_FILE'].exists():
            content = CONFIG['PICKS_FILE'].read_text(encoding='utf-8')
            # 從 markdown 中提取台股代碼（格式：XXXX.TW）
            stocks = []
            for line in content.split('\n'):
                line = line.strip()
                if '.TW' in line:
                    # 提取 XXXX.TW 格式
                    parts = line.split()
                    for part in parts:
                        if '.TW' in part:
                            stock = part.replace('`', '').strip()
                            if stock and stock not in stocks:
                                stocks.append(stock)
            logger.info(f"讀取候選股清單: {stocks}")
            return stocks
        else:
            logger.warning(f"找不到候選股清單: {CONFIG['PICKS_FILE']}")
            return []
    except Exception as e:
        logger.error(f"讀取候選股清單失敗: {e}")
        return []

def load_positions() -> Dict:
    """讀取當前持倉"""
    try:
        if CONFIG['POSITIONS_FILE'].exists():
            content = CONFIG['POSITIONS_FILE'].read_text(encoding='utf-8')
            # 簡單的 JSON 提取（假設檔案內有 JSON 塊）
            positions = {}
            for line in content.split('\n'):
                if line.strip().startswith('```json'):
                    continue
                if line.strip() == '```':
                    continue
                try:
                    data = json.loads(line)
                    positions.update(data)
                except:
                    pass
            return positions
        return {}
    except Exception as e:
        logger.error(f"讀取持倉失敗: {e}")
        return {}

def save_positions(positions: Dict):
    """保存持倉資訊"""
    try:
        CONFIG['POSITIONS_FILE'].parent.mkdir(parents=True, exist_ok=True)
        content = f"""# 當沖持倉記錄

更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 當前持倉

```json
{json.dumps(positions, ensure_ascii=False, indent=2)}
```
"""
        CONFIG['POSITIONS_FILE'].write_text(content, encoding='utf-8')
        logger.info(f"持倉已更新: {list(positions.keys())}")
    except Exception as e:
        logger.error(f"保存持倉失敗: {e}")

def append_result(stock: str, entry_price: float, exit_price: float, 
                  entry_time: str, exit_time: str, reason: str, signals: str = ""):
    """追加交易結果"""
    try:
        CONFIG['RESULTS_FILE'].parent.mkdir(parents=True, exist_ok=True)
        profit = exit_price - entry_price
        profit_pct = (profit / entry_price) * 100
        
        result_line = f"| {stock} | {entry_price:.2f} | {exit_price:.2f} | {profit:.2f} | {profit_pct:+.2f}% | {entry_time} | {exit_time} | {reason} | {signals} |\n"
        
        # 如果檔案不存在，先寫入表頭
        if not CONFIG['RESULTS_FILE'].exists():
            header = """# 當沖交易結果

| 股票代碼 | 進場價 | 出場價 | 淨利 | 淨利率 | 進場時間 | 出場時間 | 出場原因 | 觸發訊號 |
|---------|--------|--------|------|--------|---------|---------|---------|---------|
"""
            CONFIG['RESULTS_FILE'].write_text(header, encoding='utf-8')
        
        with open(CONFIG['RESULTS_FILE'], 'a', encoding='utf-8') as f:
            f.write(result_line)
        
        logger.info(f"交易結果已記錄: {stock} {profit_pct:+.2f}%")
    except Exception as e:
        logger.error(f"記錄交易結果失敗: {e}")

def write_alert(message: str):
    """寫入警報訊息"""
    try:
        CONFIG['ALERT_FILE'].parent.mkdir(parents=True, exist_ok=True)
        alert_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"
        
        with open(CONFIG['ALERT_FILE'], 'a', encoding='utf-8') as f:
            f.write(alert_msg)
        
        logger.info(f"警報已寫入: {message}")
    except Exception as e:
        logger.error(f"寫入警報失敗: {e}")

# ============================================================================
# 技術指標計算
# ============================================================================
def get_stock_data(stock: str, period: str = '1d', interval: str = '1h') -> Optional[pd.DataFrame]:
    """獲取股票數據"""
    try:
        logger.debug(f"獲取 {stock} 的 {period} {interval} 數據")
        df = yf.download(stock, period=period, interval=interval, progress=False)
        if df.empty:
            logger.warning(f"無法獲取 {stock} 的數據")
            return None
        return df
    except Exception as e:
        logger.error(f"獲取 {stock} 數據失敗: {e}")
        return None

def calculate_indicators(df: pd.DataFrame) -> Dict:
    """計算所有技術指標"""
    try:
        indicators = {}
        
        # 基礎價格和成交量
        indicators['price'] = df['Close'].iloc[-1]
        indicators['prev_price'] = df['Close'].iloc[-2] if len(df) > 1 else indicators['price']
        indicators['high'] = df['High'].iloc[-1]
        indicators['low'] = df['Low'].iloc[-1]
        indicators['volume'] = df['Volume'].iloc[-1]
        indicators['avg_volume_5d'] = df['Volume'].tail(5).mean()
        indicators['volume_ratio'] = indicators['volume'] / indicators['avg_volume_5d']
        indicators['price_change'] = ((indicators['price'] - indicators['prev_price']) / indicators['prev_price']) * 100 if indicators['prev_price'] > 0 else 0
        
        # ========== 基礎指標 ==========
        # 1. 均線
        indicators['ma5'] = df['Close'].rolling(window=5).mean().iloc[-1]
        indicators['ma10'] = df['Close'].rolling(window=10).mean().iloc[-1]
        indicators['ma20'] = df['Close'].rolling(window=20).mean().iloc[-1]
        
        # 2. EMA多重
        indicators['ema8'] = ta.trend.ema_indicator(df['Close'], window=8).iloc[-1]
        indicators['ema21'] = ta.trend.ema_indicator(df['Close'], window=21).iloc[-1]
        indicators['ema55'] = ta.trend.ema_indicator(df['Close'], window=55).iloc[-1]
        
        # 3. RSI
        indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
        
        # ========== 動能指標 ==========
        # 4. MACD
        macd = ta.trend.macd(df['Close'], window_fast=12, window_slow=26, window_sign=9)
        if macd is not None and len(macd) > 0:
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = ta.trend.macd_signal(df['Close'], window_fast=12, window_slow=26, window_sign=9).iloc[-1]
            indicators['macd_diff'] = ta.trend.macd_diff(df['Close'], window_fast=12, window_slow=26, window_sign=9).iloc[-1]
            # 檢查 MACD 上穿信號線
            if len(macd) > 1:
                indicators['macd_cross_bullish'] = (
                    indicators['macd'] > indicators['macd_signal'] and 
                    macd.iloc[-2] <= ta.trend.macd_signal(df['Close'], window_fast=12, window_slow=26, window_sign=9).iloc[-2]
                )
            else:
                indicators['macd_cross_bullish'] = False
        else:
            indicators['macd'] = 0
            indicators['macd_signal'] = 0
            indicators['macd_diff'] = 0
            indicators['macd_cross_bullish'] = False
        
        # 5. Stochastic KD
        stoch_k = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_k=3).iloc[-1]
        stoch_d = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_k=3, smooth_d=3).iloc[-1]
        indicators['stoch_k'] = stoch_k if not np.isnan(stoch_k) else 50
        indicators['stoch_d'] = stoch_d if not np.isnan(stoch_d) else 50
        
        # 6. Williams %R
        indicators['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14).iloc[-1]
        if np.isnan(indicators['williams_r']):
            indicators['williams_r'] = -50
        
        # 7. MFI (Money Flow Index)
        indicators['mfi'] = ta.momentum.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14).iloc[-1]
        if np.isnan(indicators['mfi']):
            indicators['mfi'] = 50
        
        # ========== 趨勢指標 ==========
        # 8. ADX
        adx = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        indicators['adx'] = adx.iloc[-1] if adx is not None and len(adx) > 0 else 20
        if np.isnan(indicators['adx']):
            indicators['adx'] = 20
        
        # 9. Parabolic SAR
        psar = ta.trend.psar(df['High'], df['Low'], df['Close'], iaf=0.02, maxaf=0.2)
        if psar is not None and len(psar) > 0:
            indicators['psar'] = psar.iloc[-1]
            # SAR 翻到價格下方 = 買入信號
            indicators['psar_bullish'] = indicators['psar'] < indicators['price']
        else:
            indicators['psar'] = indicators['price']
            indicators['psar_bullish'] = False
        
        # ========== 波動和通道 ==========
        # 10. ATR (Average True Range)
        indicators['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
        if np.isnan(indicators['atr']):
            indicators['atr'] = indicators['price'] * 0.02  # 預設2%
        
        # 11. 布林帶
        bb = ta.volatility.bollinger_bands(df['Close'], window=20, window_dev=2)
        if bb is not None:
            indicators['bb_high'] = bb.iloc[-1, 0] if bb.shape[1] > 0 else indicators['price']
            indicators['bb_mid'] = bb.iloc[-1, 1] if bb.shape[1] > 1 else indicators['price']
            indicators['bb_low'] = bb.iloc[-1, 2] if bb.shape[1] > 2 else indicators['price']
            # 價格觸及下軌反彈或突破上軌
            indicators['bb_bullish'] = (
                (indicators['price'] < indicators['bb_low'] and indicators['price_change'] > 0) or  # 下軌反彈
                (indicators['price'] > indicators['bb_high'])  # 突破上軌
            )
        else:
            indicators['bb_high'] = indicators['price']
            indicators['bb_mid'] = indicators['price']
            indicators['bb_low'] = indicators['price']
            indicators['bb_bullish'] = False
        
        # 12. CCI (Commodity Channel Index)
        indicators['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20).iloc[-1]
        if np.isnan(indicators['cci']):
            indicators['cci'] = 0
        
        # ========== 成交量指標 ==========
        # 13. OBV (On Balance Volume)
        obv = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        if obv is not None and len(obv) > 1:
            indicators['obv'] = obv.iloc[-1]
            # OBV 上升趨勢
            indicators['obv_bullish'] = obv.iloc[-1] > obv.iloc[-2]
        else:
            indicators['obv'] = 0
            indicators['obv_bullish'] = False
        
        # ========== 高階指標 ==========
        # 14. VWAP (Volume Weighted Average Price)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cum_tp_vol = (typical_price * df['Volume']).rolling(window=20).sum()
        cum_vol = df['Volume'].rolling(window=20).sum()
        indicators['vwap'] = (cum_tp_vol / cum_vol).iloc[-1] if cum_vol.iloc[-1] > 0 else indicators['price']
        if np.isnan(indicators['vwap']):
            indicators['vwap'] = indicators['price']
        # VWAP買入：價格在VWAP之上且回測反彈
        indicators['vwap_bullish'] = indicators['price'] > indicators['vwap']
        
        # 15. Ichimoku (一目均衡表) - 簡化版本
        # 轉換線 = (9日高點 + 9日低點) / 2
        # 基準線 = (26日高點 + 26日低點) / 2
        # 先行帶A = (轉換線 + 基準線) / 2
        # 先行帶B = (52日高點 + 52日低點) / 2
        high_9 = df['High'].rolling(window=9).max().iloc[-1]
        low_9 = df['Low'].rolling(window=9).min().iloc[-1]
        high_26 = df['High'].rolling(window=26).max().iloc[-1]
        low_26 = df['Low'].rolling(window=26).min().iloc[-1]
        high_52 = df['High'].rolling(window=52).max().iloc[-1]
        low_52 = df['Low'].rolling(window=52).min().iloc[-1]
        
        indicators['ichimoku_conversion'] = (high_9 + low_9) / 2
        indicators['ichimoku_base'] = (high_26 + low_26) / 2
        indicators['ichimoku_cloud_a'] = (indicators['ichimoku_conversion'] + indicators['ichimoku_base']) / 2
        indicators['ichimoku_cloud_b'] = (high_52 + low_52) / 2
        
        # Ichimoku 買入信號：價格在雲層之上、轉換線上穿基準線
        cloud_top = max(indicators['ichimoku_cloud_a'], indicators['ichimoku_cloud_b'])
        indicators['ichimoku_bullish'] = (
            (indicators['price'] > cloud_top) and  # 價格在雲層之上
            (indicators['ichimoku_conversion'] > indicators['ichimoku_base'])  # 轉換線上穿基準線
        )
        
        # 16. Donchian Channel (唐奇安通道)
        high_20 = df['High'].rolling(window=20).max().iloc[-1]
        low_20 = df['Low'].rolling(window=20).min().iloc[-1]
        indicators['donchian_high'] = high_20
        indicators['donchian_low'] = low_20
        # 突破20日高點買入
        indicators['donchian_bullish'] = indicators['price'] >= high_20
        
        # ========== K棒型態 ==========
        # 17. K棒型態檢測
        indicators['candle_pattern'] = detect_candle_patterns(df)
        
        # === 新增 4 個指標 (v2.0.0) ===
        # 18. Shioaji Order Flow + Delta (需要股票代碼，在 process_stock 中計算)
        indicators['order_flow'] = 0.0
        indicators['delta'] = 0.0
        
        # 19. FinMind 三大法人買賣超 (需要股票代碼，在 process_stock 中計算)
        indicators['institutional'] = 0.0
        
        # 20. 期交所 Put/Call Ratio (全市場指標，取一次快取)
        indicators['put_call_ratio'] = 0.0
        
        return indicators
    except Exception as e:
        logger.error(f"計算指標失敗: {e}")
        logger.debug(traceback.format_exc())
        return {}

def detect_candle_patterns(df: pd.DataFrame) -> Dict:
    """檢測K棒型態"""
    patterns = {
        'hammer': False,
        'engulfing': False,
        'doji': False,
        'morning_star': False,
        'score': 0
    }
    
    try:
        if len(df) < 3:
            return patterns
        
        # 最新的K棒
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev_prev = df.iloc[-3]
        
        open_c = current['Open']
        close_c = current['Close']
        high_c = current['High']
        low_c = current['Low']
        
        open_p = prev['Open']
        close_p = prev['Close']
        
        open_pp = prev_prev['Open']
        close_pp = prev_prev['Close']
        
        # 錘子線 (Hammer): 下影線 > 2倍實體，上影線很短，實體在上半部
        body = abs(close_c - open_c)
        lower_shadow = min(open_c, close_c) - low_c
        upper_shadow = high_c - max(open_c, close_c)
        
        if body > 0:
            if lower_shadow > 2 * body and upper_shadow < body * 0.3 and close_c > open_c:
                patterns['hammer'] = True
                patterns['score'] += 5
        
        # 吞噬型態 (Engulfing)
        if close_c > open_p and open_c < close_p:  # 看漲吞噬
            if (close_c - open_c) > (close_p - open_p):
                patterns['engulfing'] = True
                patterns['score'] += 5
        
        # 十字星 (Doji)
        if body < (high_c - low_c) * 0.1:  # 實體非常小
            patterns['doji'] = True
            patterns['score'] += 3
        
        # 晨星 (Morning Star): 三根K棒組合
        prev_prev_close = close_pp
        prev_prev_open = open_pp
        prev_close = close_p
        prev_open = open_p
        
        # 下跌 -> 小實體 -> 上漲
        if (prev_prev_close < prev_prev_open and  # 前前根下跌
            (abs(close_p - open_p) < abs(close_pp - open_pp) * 0.5) and  # 前根實體小
            close_c > open_c):  # 當前根上漲
            patterns['morning_star'] = True
            patterns['score'] += 5
        
        return patterns
    except Exception as e:
        logger.error(f"K棒型態檢測失敗: {e}")
        return patterns

# ============================================================================
# 訊號評分系統
# ============================================================================
def calculate_buy_score(indicators: Dict, stock_symbol: str = "") -> Tuple[int, List[str], float]:
    """
    計算買入評分
    
    Args:
        indicators: 技術指標字典
        stock_symbol: 股票代碼（用於查詢新聞情緒）
    
    Returns:
        (評分, 訊號列表, 新聞情緒分數)
    """
    score = 0
    signals = []
    news_sentiment = 0.0
    
    try:
        # 0. 新聞情緒分析 (先查詢，方便稍後使用)
        if stock_symbol:
            news_sentiment = get_news_sentiment(stock_symbol)
            if news_sentiment > 0.3:
                score += SIGNAL_WEIGHTS['news_sentiment']
                signals.append(f"新聞情緒({news_sentiment:.3f})")
                # 額外獎勵：情緒 > 0.7
                if news_sentiment > 0.7:
                    score += 5  # Bonus 5分
                    signals[-1] = f"新聞情緒強勢({news_sentiment:.3f})"
        
        # 1. 均線交叉 (5MA > 10MA)
        if indicators.get('ma5', 0) > indicators.get('ma10', 0):
            score += SIGNAL_WEIGHTS['ma_cross']
            signals.append(f"均線交叉(MA5>MA10)")
        
        # 2. RSI 超賣/強勢
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score += SIGNAL_WEIGHTS['rsi']
            signals.append(f"RSI超賣({rsi:.1f})")
        elif rsi > 50:
            score += SIGNAL_WEIGHTS['rsi'] * 0.8  # 8分的80% = 6.4
            signals.append(f"RSI強勢({rsi:.1f})")
        
        # 3. 成交量爆發
        volume_ratio = indicators.get('volume_ratio', 0)
        if volume_ratio >= CONFIG['VOLUME_MULTIPLIER']:
            score += SIGNAL_WEIGHTS['volume']
            signals.append(f"成交量爆發({volume_ratio:.2f}x)")
        
        # 4. VWAP 買入信號
        if indicators.get('vwap_bullish', False):
            score += SIGNAL_WEIGHTS['vwap']
            signals.append("VWAP買入")
        
        # 5. MACD 交叉
        if indicators.get('macd_cross_bullish', False):
            score += SIGNAL_WEIGHTS['macd']
            signals.append("MACD上穿")
        elif indicators.get('macd', 0) > indicators.get('macd_signal', 0):
            score += SIGNAL_WEIGHTS['macd'] * 0.5  # 部分credit
            signals.append(f"MACD正值")
        
        # 6. 布林帶買入信號
        if indicators.get('bb_bullish', False):
            score += SIGNAL_WEIGHTS['bollinger']
            signals.append("布林帶突破")
        
        # 7. K棒型態
        candle_score = indicators.get('candle_pattern', {}).get('score', 0)
        if candle_score > 0:
            score += min(candle_score, SIGNAL_WEIGHTS['candle_pattern'])
            patterns = indicators.get('candle_pattern', {})
            if patterns.get('hammer'):
                signals.append("錘子線")
            if patterns.get('engulfing'):
                signals.append("吞噬型態")
            if patterns.get('doji'):
                signals.append("十字星")
            if patterns.get('morning_star'):
                signals.append("晨星")
        
        # 8. Stochastic KD 買入信號
        stoch_k = indicators.get('stoch_k', 50)
        if stoch_k < 20 and stoch_k > indicators.get('stoch_d', 50):  # K < 20且上穿D
            score += SIGNAL_WEIGHTS['stochastic']
            signals.append(f"StochKD超賣反彈({stoch_k:.1f})")
        
        # 9. ADX 趨勢強度
        adx = indicators.get('adx', 20)
        if adx > 25:
            score += SIGNAL_WEIGHTS['adx']
            signals.append(f"ADX趨勢({adx:.1f})")
        
        # 10. CCI 買入信號
        cci = indicators.get('cci', 0)
        if cci < -100 or (cci > -100 and cci < 0):  # CCI從-100以下回升
            score += SIGNAL_WEIGHTS['cci']
            signals.append(f"CCI反彈({cci:.1f})")
        
        # 11. Williams %R 買入信號
        williams = indicators.get('williams_r', -50)
        if williams < -80:  # 超賣區
            score += SIGNAL_WEIGHTS['williams']
            signals.append(f"Williams超賣({williams:.1f})")
        
        # 12. OBV 買入信號
        if indicators.get('obv_bullish', False):
            score += SIGNAL_WEIGHTS['obv']
            signals.append("OBV上升")
        
        # 13. MFI 買入信號
        mfi = indicators.get('mfi', 50)
        if mfi < 30:  # 超賣區
            score += SIGNAL_WEIGHTS['mfi']
            signals.append(f"MFI超賣({mfi:.1f})")
        
        # 14. Parabolic SAR
        if indicators.get('psar_bullish', False):
            score += SIGNAL_WEIGHTS['psar']
            signals.append("SAR翻正")
        
        # 15. Ichimoku
        if indicators.get('ichimoku_bullish', False):
            score += SIGNAL_WEIGHTS['ichimoku']
            signals.append("Ichimoku買入")
        
        # 16. Donchian Channel 突破
        if indicators.get('donchian_bullish', False):
            score += SIGNAL_WEIGHTS['donchian']
            signals.append("Donchian突破")
        
        # 17. EMA 多重確認
        if (indicators.get('ema8', 0) > indicators.get('ema21', 0) and 
            indicators.get('ema21', 0) > indicators.get('ema55', 0)):
            score += SIGNAL_WEIGHTS['ema_multi']
            signals.append("EMA多重確認")
        
        # === 新增 4 個指標 (v2.0.0) ===
        
        # 18. Shioaji Order Flow (買方掛單 > 賣方掛單)
        order_flow_score = indicators.get('order_flow', 0.0)
        if order_flow_score > 5:  # 若 order flow 評分 > 5，加分
            score += min(SIGNAL_WEIGHTS['order_flow'], order_flow_score)
            signals.append(f"掛單流量({order_flow_score:.1f})")
        
        # 19. Shioaji Delta (累積 delta 為正)
        delta_score = indicators.get('delta', 0.0)
        if delta_score > 5:
            score += min(SIGNAL_WEIGHTS['delta'], delta_score)
            signals.append(f"成交Delta({delta_score:.1f})")
        
        # 20. FinMind 三大法人買賣超 (連續買超)
        institutional_score = indicators.get('institutional', 0.0)
        if institutional_score > 0:
            score += min(SIGNAL_WEIGHTS['institutional'], institutional_score)
            signals.append(f"法人買超({institutional_score:.1f})")
        
        # 21. 期交所 Put/Call Ratio (偏多訊號)
        put_call_score = indicators.get('put_call_ratio', 0.0)
        if put_call_score > 0:
            score += min(SIGNAL_WEIGHTS['put_call_ratio'], put_call_score)
            signals.append(f"P/C偏多({put_call_score:.1f})")
        
        return score, signals, news_sentiment
    except Exception as e:
        logger.error(f"計算買入評分失敗: {e}")
        return 0, [], 0.0

def generate_buy_signal(indicators: Dict, stock_symbol: str = "") -> Tuple[bool, str, int, str, float]:
    """
    根據評分制生成買入訊號
    
    Args:
        indicators: 技術指標字典
        stock_symbol: 股票代碼（用於新聞情緒分析）
    
    Returns:
        (是否買入, 訊號字符串, 評分, 評分說明, 新聞情緒分數)
    """
    score, signals, news_sentiment = calculate_buy_score(indicators, stock_symbol)
    signal_str = " + ".join(signals) if signals else "無訊號"
    
    if score >= CONFIG['BUY_SCORE_THRESHOLD']:
        return True, signal_str, score, f"總分{score}/100", news_sentiment
    else:
        return False, signal_str, score, f"總分{score}/100 (需{CONFIG['BUY_SCORE_THRESHOLD']})", news_sentiment

def generate_sell_signal(current_price: float, entry_price: float, atr: float,
                        entry_time: datetime) -> Tuple[bool, str]:
    """根據動態ATR生成賣出訊號"""
    now = datetime.now()
    time_held = (now - entry_time).total_seconds() / 60  # 分鐘
    
    # 計算動態停損停利
    stop_loss_price = entry_price - (2 * atr)
    take_profit_price = entry_price + (3 * atr)
    
    # 檢查停損
    if current_price <= stop_loss_price:
        loss_pct = (current_price - entry_price) / entry_price
        return True, f"動態停損 ({loss_pct*100:.2f}%) [ATR: {atr:.2f}]"
    
    # 檢查停利
    if current_price >= take_profit_price:
        profit_pct = (current_price - entry_price) / entry_price
        return True, f"動態停利 ({profit_pct*100:.2f}%) [ATR: {atr:.2f}]"
    
    # 檢查時間限制：13:25 強制平倉
    if now.hour == CONFIG['FORCE_CLOSE_HOUR'] and now.minute >= CONFIG['FORCE_CLOSE_MINUTE']:
        profit_pct = (current_price - entry_price) / entry_price
        return True, f"強制平倉 13:25 ({profit_pct*100:.2f}%)"
    
    return False, "持倉中"

# ============================================================================
# 主交易引擎
# ============================================================================
class DaytradeBot:
    def __init__(self):
        self.positions = load_positions()
        self.running = True
    
    def is_trading_time(self) -> bool:
        """檢查是否在交易時段內"""
        now = datetime.now()
        start = dt_time(CONFIG['START_HOUR'], 0)
        end = dt_time(CONFIG['END_HOUR'], CONFIG['END_MINUTE'])
        return start <= now.time() <= end
    
    def can_open_position(self) -> bool:
        """檢查是否還能開新倉"""
        now = datetime.now()
        cutoff = dt_time(CONFIG['NO_OPEN_AFTER_HOUR'], CONFIG['NO_OPEN_AFTER_MINUTE'])
        return now.time() < cutoff
    
    def process_stock(self, stock: str):
        """處理單隻股票"""
        try:
            # 獲取數據
            df = get_stock_data(stock, period='5d', interval='1h')
            if df is None or df.empty:
                logger.warning(f"{stock} 無可用數據，跳過")
                return
            
            # 計算指標
            indicators = calculate_indicators(df)
            if not indicators:
                logger.warning(f"{stock} 指標計算失敗，跳過")
                return
            
            # === 新增 4 個指標的動態計算 (v2.0.0) ===
            # Shioaji Order Flow + Delta
            try:
                order_flow_score, delta_score = get_shioaji_order_flow(stock)
                indicators['order_flow'] = order_flow_score
                indicators['delta'] = delta_score
            except Exception as e:
                logger.debug(f"{stock} Shioaji 計算失敗: {e}")
            
            # FinMind 三大法人買賣超
            try:
                institutional_score = get_finmind_institutional(stock)
                indicators['institutional'] = institutional_score
            except Exception as e:
                logger.debug(f"{stock} FinMind 計算失敗: {e}")
            
            # 期交所 Put/Call Ratio (全市場指標，每支股票共用)
            try:
                put_call_score = get_taifex_put_call_ratio()
                indicators['put_call_ratio'] = put_call_score
            except Exception as e:
                logger.debug(f"期交所 P/C Ratio 計算失敗: {e}")
            
            current_price = indicators['price']
            atr = indicators['atr']
            
            logger.debug(f"{stock}: 價格={current_price:.2f}, ATR={atr:.4f}, RSI={indicators.get('rsi', 0):.1f}")
            
            # 檢查持倉的出場訊號
            if stock in self.positions:
                pos = self.positions[stock]
                should_sell, reason = generate_sell_signal(
                    current_price, 
                    pos['entry_price'],
                    atr,
                    datetime.fromisoformat(pos['entry_time'])
                )
                
                if should_sell:
                    exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    profit_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    
                    # 記錄交易結果
                    append_result(
                        stock, 
                        pos['entry_price'], 
                        current_price, 
                        pos['entry_time'], 
                        exit_time, 
                        reason,
                        pos.get('signal', '')
                    )
                    
                    # 寫入警報（包含新聞情緒分數）
                    news_sentiment = pos.get('news_sentiment', 0.0)
                    sentiment_info = f" | 新聞情緒:{news_sentiment:.3f}" if news_sentiment > 0 else ""
                    alert_msg = f"【賣出】{stock} @ {current_price:.2f} | {reason} | 損益: {profit_pct:+.2f}%{sentiment_info}"
                    write_alert(alert_msg)
                    
                    # 移除持倉
                    del self.positions[stock]
                    save_positions(self.positions)
                    logger.info(f"【賣出】{stock} @ {current_price:.2f} - {reason}{sentiment_info}")
            else:
                # 檢查買入訊號（只在可開倉時段）
                if self.can_open_position():
                    should_buy, signal_str, score, score_msg, news_sentiment = generate_buy_signal(indicators, stock)
                    
                    if should_buy:
                        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # 新增持倉
                        self.positions[stock] = {
                            'entry_price': current_price,
                            'entry_time': entry_time,
                            'signal': signal_str,
                            'score': score,
                            'news_sentiment': news_sentiment
                        }
                        save_positions(self.positions)
                        
                        # 寫入警報（包含新聞情緒分數）
                        sentiment_info = f" | 新聞情緒:{news_sentiment:.3f}" if news_sentiment > 0 else ""
                        alert_msg = f"【買入】{stock} @ {current_price:.2f} | {signal_str} | {score_msg}{sentiment_info}"
                        write_alert(alert_msg)
                        
                        logger.info(f"【買入】{stock} @ {current_price:.2f} - {signal_str} ({score_msg}){sentiment_info}")
                    else:
                        logger.debug(f"{stock} 無買入信號 ({score_msg})")
        
        except Exception as e:
            logger.error(f"處理 {stock} 時發生錯誤: {e}")
            logger.debug(traceback.format_exc())
    
    def run(self):
        """主運行迴圈"""
        logger.info("=" * 80)
        logger.info(f"台股當沖模擬交易 Bot v{__version__} - 22+指標評分系統")
        logger.info(f"新增指標: Shioaji Order Flow + Delta | FinMind 法人買賣超 | 期交所 P/C Ratio")
        logger.info(f"買入門檻: {CONFIG['BUY_SCORE_THRESHOLD']}/100 分")
        logger.info(f"動態停損: 買入價 - 2*ATR")
        logger.info(f"動態停利: 買入價 + 3*ATR")
        logger.info(f"掃描間隔: {CONFIG['SCAN_INTERVAL']}秒（無持倉）/ {CONFIG['MONITOR_INTERVAL']}秒（有持倉）")
        logger.info("=" * 80)
        
        try:
            while self.running:
                try:
                    # 根據是否有持倉決定檢查頻率
                    has_positions = len(self.positions) > 0
                    interval = CONFIG['MONITOR_INTERVAL'] if has_positions else CONFIG['SCAN_INTERVAL']
                    
                    # 檢查是否在交易時段
                    if not self.is_trading_time():
                        now = datetime.now().strftime('%H:%M:%S')
                        logger.debug(f"({now}) 非交易時段，等待中...")
                        time.sleep(CONFIG['SCAN_INTERVAL'])
                        continue
                    
                    # 讀取候選股清單
                    picks = read_picks()
                    if not picks:
                        logger.warning("沒有候選股清單，跳過本輪檢查")
                        time.sleep(CONFIG['SCAN_INTERVAL'])
                        continue
                    
                    logger.info(f"掃描 {len(picks)} 檔（間隔 {interval}s）({datetime.now().strftime('%H:%M:%S')})")
                    
                    # 處理每隻股票
                    for stock in picks:
                        self.process_stock(stock)
                        time.sleep(0.5)  # 每檔間隔 500ms，避免 rate limit
                    
                    logger.info(f"本輪完成，持倉: {len(self.positions)} 檔 ({list(self.positions.keys())})")
                    
                    # 根據持倉狀態調整等待時間
                    time.sleep(interval)
                
                except KeyboardInterrupt:
                    logger.info("收到停止信號")
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"主迴圈異常: {e}")
                    logger.debug(traceback.format_exc())
                    time.sleep(CONFIG['SCAN_INTERVAL'])
        
        finally:
            logger.info("Bot 已停止")
            self.save_final_state()
    
    def save_final_state(self):
        """保存最終狀態"""
        try:
            save_positions(self.positions)
            logger.info(f"最終狀態已保存，當前持倉: {self.positions}")
        except Exception as e:
            logger.error(f"保存最終狀態失敗: {e}")

# ============================================================================
# 入口
# ============================================================================
if __name__ == '__main__':
    try:
        bot = DaytradeBot()
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
