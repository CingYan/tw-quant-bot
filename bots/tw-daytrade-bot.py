#!/usr/bin/env -S uv run --python 3.13 python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "shioaji>=1.3",
#   "yfinance>=0.2",
#   "requests>=2.28",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "ta>=0.11",
# ]
# ///
# -*- coding: utf-8 -*-
"""
台股當沖模擬交易 Bot — Shioaji + SMC + VWAP
版本: 1.0.0

架構（基於美股 bot 移植）：
  - Shioaji API 即時報價（全域初始化，simulation=True）
  - yfinance K 線資料（加 .TW 後綴）
  - SMC Engine：OB / FVG / BOS / CHoCH / Sweep
  - VWAPCalculator：日內累積式 VWAP + ±1σ/±2σ 偏差帶 + Bounce 偵測
  - DailyRiskManager：日損上限 1.5%、連虧 3 筆鎖倉 + 30 分鐘自動解鎖
  - Kelly × 0.3 部位計算
  - 移動停損（trailing stop）
  - 整張交易（1000 股為單位，不做零股）

交易時段（台灣時間 UTC+8）：
  - Kill Zone 1: 09:00–10:00（開盤）
  - Kill Zone 2: 12:30–13:20（尾盤）
  - 強制平倉: 13:20
  - 不開新倉: 13:00 之後

⚠️ 模擬交易專用 — simulation=True，禁止真實下單
⚠️ API Key 一律從 credentials JSON 載入，禁止 hardcode
"""

__version__ = "1.0.0"
__author__ = "tw-daytrade-bot"

import sys
import os
import json
import logging
import time
import traceback
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime, date, time as dt_time, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ============================================================================
# 依賴自動安裝
# ============================================================================

def check_and_install_dependencies() -> None:
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yfinance': 'yfinance',
        'ta': 'ta',
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] 安裝缺失套件: {missing}")
        try:
            import shutil
            uv_path = shutil.which('uv')
            if uv_path:
                subprocess.check_call([uv_path, 'pip', 'install', '-q'] + missing)
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-q'] + missing)
        print("[SETUP] 安裝完成")

check_and_install_dependencies()

import pandas as pd
import numpy as np
import yfinance as yf
import ta

# ============================================================================
# Telegram 通知
# ============================================================================

def _tg_notify(text: str, chat_id: str = os.environ.get('TG_CHAT_ID', '')) -> None:
    """發送 Telegram 通知（靜默失敗，不影響交易）"""
    try:
        config_path = Path('/home/node/.openclaw/openclaw.json')
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text())
        # channels 是 dict（如 {"telegram": {...}}），不是 list
        channels = config.get('channels', {})
        token = None
        if isinstance(channels, dict):
            tg = channels.get('telegram', {})
            if isinstance(tg, dict):
                token = tg.get('botToken') or tg.get('token')
        elif isinstance(channels, list):
            for ch in channels:
                if isinstance(ch, dict) and ch.get('type') == 'telegram':
                    token = ch.get('botToken') or ch.get('token')
                    break
        if not token:
            return
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'HTML',
            'disable_notification': 'false',
        }).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception:
        pass

# ============================================================================
# 路徑配置
# ============================================================================

BASE_DIR = Path('/home/node/clawd')
CRED_FILE = BASE_DIR / 'credentials' / 'shioaji-credentials.json'
DATA_DIR = BASE_DIR / 'data' / 'tw-daytrade'
DATA_DIR.mkdir(parents=True, exist_ok=True)

TW_TZ = timezone(timedelta(hours=8))

_FAKE_TIME: Optional[datetime] = None  # 測試用：設定後覆蓋系統時間

def _tw_now() -> datetime:
    if _FAKE_TIME is not None:
        return _FAKE_TIME
    return datetime.now(TW_TZ)

def _tw_date_str() -> str:
    return _tw_now().strftime('%Y-%m-%d')

def trade_log_path() -> Path:
    return DATA_DIR / f'trades-{_tw_date_str()}.json'

def bot_log_path() -> Path:
    return DATA_DIR / f'bot-{_tw_date_str()}.log'

def positions_path() -> Path:
    return DATA_DIR / 'positions.json'

# ============================================================================
# 日誌系統
# ============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger('tw_daytrade_bot')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(bot_log_path(), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

logger = setup_logging()

# ============================================================================
# Credentials 載入
# ============================================================================

def load_credentials() -> Dict:
    try:
        with open(CRED_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.critical(f"找不到 credentials 檔案: {CRED_FILE}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.critical(f"credentials 解析失敗: {e}")
        sys.exit(1)

_CREDS = load_credentials()
_SHIOAJI = _CREDS.get('shioaji', {})

SHIOAJI_API_KEY: str = _SHIOAJI.get('api_key', '')
SHIOAJI_SECRET_KEY: str = _SHIOAJI.get('secret_key', '')

if not SHIOAJI_API_KEY or not SHIOAJI_SECRET_KEY:
    logger.critical("Shioaji API Key / Secret 未設定")
    sys.exit(1)

# ============================================================================
# Shioaji 全域初始化（只登入一次）
# ============================================================================

_shioaji_api = None

def get_shioaji_api():
    """取得全域 Shioaji API 實例（延遲初始化，只登入一次）"""
    global _shioaji_api
    if _shioaji_api is not None:
        return _shioaji_api
    try:
        import shioaji as sj
        api = sj.Shioaji(simulation=True)
        api.login(
            api_key=SHIOAJI_API_KEY,
            secret_key=SHIOAJI_SECRET_KEY,
            contracts_cb=lambda security_type: None,
        )
        logger.info("Shioaji 登入成功（simulation=True）")
        _shioaji_api = api
        return api
    except Exception as e:
        logger.error(f"Shioaji 登入失敗: {e}")
        return None


# 2026-03-11 升級：全域即時報價快取（subscribe 事件驅動）
# ============================================================================
_live_prices: Dict[str, float] = {}          # symbol → 最新成交價
_live_prices_ts: Dict[str, float] = {}       # symbol → 更新 timestamp
_subscribed_symbols: set = set()

def _on_tick_stk_v1(exchange, tick):
    """Tick callback：更新全域即時價格"""
    code = getattr(tick, 'code', '')
    price = getattr(tick, 'close', 0.0)
    if code and price > 0:
        _live_prices[code] = float(price)
        _live_prices_ts[code] = time.time()

def subscribe_quotes(symbols: List[str]) -> None:
    """訂閱即時報價（最多 200 支）"""
    api = get_shioaji_api()
    if api is None:
        logger.warning("Shioaji 未連線，無法訂閱報價")
        return
    import shioaji as sj
    api.quote.set_on_tick_stk_v1_callback(_on_tick_stk_v1)
    for sym in symbols:
        if sym in _subscribed_symbols:
            continue
        try:
            contract = api.Contracts.Stocks.get(sym)
            if contract is None:
                contract = api.Contracts.Stocks[sym]
            if contract:
                api.quote.subscribe(
                    contract,
                    quote_type=sj.constant.QuoteType.Tick,
                    version=sj.constant.QuoteVersion.v1,
                )
                _subscribed_symbols.add(sym)
        except Exception as e:
            logger.debug(f"{sym} 訂閱失敗: {e}")
    logger.info(f"📡 已訂閱 {len(_subscribed_symbols)} 支即時報價")

def unsubscribe_all() -> None:
    """取消所有報價訂閱"""
    api = get_shioaji_api()
    if api is None:
        return
    import shioaji as sj
    for sym in list(_subscribed_symbols):
        try:
            contract = api.Contracts.Stocks.get(sym)
            if contract is None:
                contract = api.Contracts.Stocks[sym]
            if contract:
                api.quote.unsubscribe(
                    contract,
                    quote_type=sj.constant.QuoteType.Tick,
                    version=sj.constant.QuoteVersion.v1,
                )
        except Exception:
            pass
    _subscribed_symbols.clear()
    logger.info("📡 已取消所有報價訂閱")

def get_live_price(symbol: str) -> float:
    """取得即時價格：優先 subscribe 快取 → Shioaji snapshot → yfinance"""
    # 1. subscribe 快取（5 秒內有效）
    if symbol in _live_prices:
        age = time.time() - _live_prices_ts.get(symbol, 0)
        if age < 5.0:
            return _live_prices[symbol]
    # 2. Shioaji snapshot fallback
    price = get_latest_price_shioaji(symbol)
    if price > 0:
        _live_prices[symbol] = price
        _live_prices_ts[symbol] = time.time()
        return price
    # 3. yfinance fallback
    df = get_kbars(symbol, interval='5m', period='1d')
    if not df.empty:
        return float(df['Close'].iloc[-1])
    return 0.0


# 2026-03-11 升級：Shioaji 下單工具函數
# ============================================================================

def shioaji_place_order(symbol: str, action: str, qty_shares: int) -> Optional[Dict]:
    """
    透過 Shioaji 下市價單（模擬模式）
    
    Args:
        symbol: 股票代碼
        action: 'buy' or 'sell'
        qty_shares: 股數（會自動轉換為張數，不足 1 張跳過）
    
    Returns:
        dict with order info, or None if failed
    """
    api = get_shioaji_api()
    if api is None:
        logger.error(f"Shioaji 未連線，無法下單 {symbol}")
        return None
    
    import shioaji as sj
    
    lots = qty_shares // 1000  # 轉換為張數
    if lots <= 0:
        logger.warning(f"{symbol} 股數 {qty_shares} 不足 1 張（1000 股），跳過下單")
        return None
    
    try:
        contract = api.Contracts.Stocks.get(symbol)
        if contract is None:
            contract = api.Contracts.Stocks[symbol]
        if contract is None:
            logger.error(f"{symbol} 找不到合約")
            return None
        
        order = api.Order(
            price=0,  # 市價單
            quantity=lots,
            action=sj.constant.Action.Buy if action == 'buy' else sj.constant.Action.Sell,
            price_type=sj.constant.StockPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
            order_lot=sj.constant.StockOrderLot.Common,
            account=api.stock_account,
        )
        
        trade = api.place_order(contract, order)
        
        # 取得訂單資訊
        order_id = getattr(trade.status, 'id', '') if trade.status else ''
        status = str(getattr(trade.status, 'status', 'Unknown'))
        
        # 2026-03-12 修復：不用 trade.status.order_quantity 判斷是否被拒
        # 原因：place_order() 返回時 status callback 可能尚未到達，
        #       order_quantity 初始值為 0，但 Shioaji 實際已接受訂單
        # 改用：trade.order.quantity（我們送出的委託量）+ 等 update_status 後再判斷
        #
        # 真正的拒絕判斷：檢查 op_code 是否為錯誤碼（如 '88' 餘股不足）
        # op_code '00' = 正常, 其他 = 異常
        op_code = ''
        if hasattr(trade, 'status') and trade.status:
            # OrderState callback 裡的 operation.op_code
            # 但 trade 物件本身不一定有 operation 屬性
            # 需要等 update_status 才能拿到最新狀態
            pass
        
        # 先等 0.5 秒讓 callback 回來，再檢查
        time.sleep(0.5)
        api.update_status(api.stock_account)
        updated_status = str(getattr(trade.status, 'status', ''))
        updated_qty = getattr(trade.status, 'order_quantity', lots) if trade.status else lots
        
        # 只在明確拒絕（Failed/Cancelled 且 order_quantity=0）時才判定失敗
        if updated_qty == 0 and ('Failed' in updated_status or 'Cancel' in updated_status):
            logger.warning(f"{symbol} 下單被拒絕（status={updated_status}, order_quantity=0）")
            return None
        
        logger.info(
            f"{'📈 買入' if action == 'buy' else '📉 賣出'} 下單: "
            f"{symbol} {lots}張 | 訂單ID: {order_id} | 狀態: {status}"
        )
        
        # 等待成交確認（最多 5 秒）
        filled_price = 0.0
        for _ in range(10):
            time.sleep(0.5)
            api.update_status(api.stock_account)
            s = trade.status
            st = str(getattr(s, 'status', ''))
            if 'Filled' in st or 'Deal' in st:
                # 嘗試從 status 取得成交價
                filled_price = float(getattr(s, 'deal_price', 0) or 0)
                if filled_price <= 0:
                    filled_price = float(getattr(s, 'modified_price', 0) or 0)
                break
            if 'Cancel' in st or 'Failed' in st:
                logger.warning(f"{symbol} 訂單被取消或失敗: {st}")
                return None
        
        # 如果無法從 status 取得成交價，用 snapshot
        if filled_price <= 0:
            filled_price = get_live_price(symbol)
        
        return {
            'order_id': order_id,
            'status': str(getattr(trade.status, 'status', '')),
            'filled_price': filled_price,
            'lots': lots,
            'qty_shares': lots * 1000,
            'trade_obj': trade,
        }
    
    except Exception as e:
        logger.error(f"{symbol} 下單失敗: {e}")
        return None


def shioaji_check_usage() -> None:
    """檢查 API 流量使用率"""
    api = get_shioaji_api()
    if api is None:
        return
    try:
        u = api.usage()
        pct = (u.bytes / u.limit_bytes * 100) if u.limit_bytes > 0 else 0
        if pct > 80:
            logger.warning(f"⚠️ API 流量使用率 {pct:.1f}%（{u.bytes/1024/1024:.1f}MB / {u.limit_bytes/1024/1024:.0f}MB）")
            _tg_notify(f"⚠️ Shioaji API 流量 {pct:.1f}%，接近上限")
        elif pct > 50:
            logger.info(f"API 流量: {pct:.1f}%")
    except Exception as e:
        logger.debug(f"流量查詢失敗: {e}")


# ============================================================================
# 交易配置
# ============================================================================

CONFIG = {
    'ATR_STOP_MULT': 2.5,
    'ATR_TP_MULT': 3.5,
    'MAX_POSITIONS': 3,
    'DAILY_LOSS_LIMIT': 0.015,
    'MAX_CONSECUTIVE_LOSSES': 3,
    'COOLDOWN_SEC': 1800,
    'ENTRY_COOLDOWN_SEC': 120,
    'KELLY_FRACTION': 0.3,
    'MAX_SINGLE_RISK_PCT': 0.005,
    'SCAN_INTERVAL_SEC': 30,
    'MONITOR_INTERVAL_SEC': 20,
    'MIN_SCORE': 70,
    'KZ1_MIN_SCORE': 90,  # 2026-03-05 教訓：KZ1 提高到 90（飽和 FVG 易在開盤進場陷阱）
    'FVG_SATURATION_THRESHOLD': 7,  # FVG 飽和度：≥7 時拒絕進場（2026-03-05）

    # 台股交易時段
    'MARKET_OPEN': dt_time(9, 0),
    'MARKET_CLOSE': dt_time(13, 30),
    'KZ1_START': dt_time(9, 0),
    'KZ1_END': dt_time(10, 0),
    'KZ2_START': dt_time(12, 30),
    'KZ2_END': dt_time(13, 20),
    'NO_NEW_TRADE_AFTER': dt_time(13, 0),
    'FORCE_CLOSE': dt_time(13, 20),

    # 漲跌停
    'LIMIT_PCT': 0.10,

    # 模擬帳戶初始資金
    'INITIAL_BALANCE': 1_000_000.0,

    # 股價上限（整張交易：price × 1000 ≤ balance × 10%）
    'MAX_STOCK_PRICE': 100.0,
}

# 預設股票池（高流動性台股）
DEFAULT_WATCHLIST = [
    # 低價高量（股價 ≤100 元，日均量大，適合整張當沖）
    '2303',  # 聯電 ~63
    '2881',  # 富邦金 ~95
    '2882',  # 國泰金 ~79
    '2891',  # 中信金 ~53
    '2886',  # 兆豐金 ~40
    '2884',  # 玉山金 ~34
    '6505',  # 台塑化 ~51
    '1301',  # 台塑 ~47
    '1303',  # 南亞 ~83
    '2002',  # 中鋼 ~20
    '3481',  # 群創 ~21
    '2409',  # 友達 ~15
    '2603',  # 長榮 ~55
    '2609',  # 陽明 ~35
    '2615',  # 萬海 ~45
    '5880',  # 合庫金 ~30
    '2890',  # 永豐金 ~20
    '2883',  # 開發金 ~17
    '1326',  # 台化 ~40
    '2912',  # 統一超 ~95 (pending filter)
]

# ============================================================================
# SMC Engine
# ============================================================================

class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str

@dataclass
class FVG:
    type: str
    top: float
    bottom: float
    ce: float
    index: int
    mitigated: bool = False

@dataclass
class OrderBlock:
    type: str
    high: float
    low: float
    index: int
    mitigated: bool = False

@dataclass
class StructureEvent:
    type: str
    direction: str
    index: int
    price: float
    swing_ref: float


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    highs = df['High'].values
    lows = df['Low'].values
    for i in range(lookback, len(df) - lookback):
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(index=i, price=float(highs[i]), type='high'))
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(index=i, price=float(lows[i]), type='low'))
    return sorted(swings, key=lambda s: s.index)


def detect_structure(df: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureEvent]:
    events: List[StructureEvent] = []
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    trend = Bias.NEUTRAL
    swing_highs = [s for s in swings if s.type == 'high']
    swing_lows = [s for s in swings if s.type == 'low']

    for i in range(1, len(df)):
        for sh in swing_highs:
            if sh.index >= i:
                continue
            if highs[i] > sh.price and closes[i] < sh.price:
                events.append(StructureEvent('sweep', 'bearish', i, float(highs[i]), sh.price))
            elif closes[i] > sh.price:
                etype = 'choch' if trend == Bias.BEARISH else 'bos'
                events.append(StructureEvent(etype, 'bullish', i, float(closes[i]), sh.price))
                trend = Bias.BULLISH
        for sl in swing_lows:
            if sl.index >= i:
                continue
            if lows[i] < sl.price and closes[i] > sl.price:
                events.append(StructureEvent('sweep', 'bullish', i, float(lows[i]), sl.price))
            elif closes[i] < sl.price:
                etype = 'choch' if trend == Bias.BULLISH else 'bos'
                events.append(StructureEvent(etype, 'bearish', i, float(closes[i]), sl.price))
                trend = Bias.BEARISH
    return events


def detect_fvg(df: pd.DataFrame) -> List[FVG]:
    fvgs: List[FVG] = []
    highs = df['High'].values
    lows = df['Low'].values
    for i in range(2, len(df)):
        if lows[i] > highs[i - 2]:
            fvgs.append(FVG('bullish', float(lows[i]), float(highs[i - 2]),
                            (float(lows[i]) + float(highs[i - 2])) / 2, i))
        if highs[i] < lows[i - 2]:
            fvgs.append(FVG('bearish', float(lows[i - 2]), float(highs[i]),
                            (float(lows[i - 2]) + float(highs[i])) / 2, i))
    return fvgs


def check_fvg_mitigation(fvgs: List[FVG], df: pd.DataFrame) -> List[FVG]:
    closes = df['Close'].values
    for fvg in fvgs:
        for i in range(fvg.index + 1, len(df)):
            if fvg.type == 'bullish' and closes[i] < fvg.bottom:
                fvg.mitigated = True; break
            elif fvg.type == 'bearish' and closes[i] > fvg.top:
                fvg.mitigated = True; break
    return fvgs


def detect_order_blocks(df: pd.DataFrame, events: List[StructureEvent]) -> List[OrderBlock]:
    obs: List[OrderBlock] = []
    opens = df['Open'].values
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    for ev in events:
        if ev.type not in ('bos', 'choch'):
            continue
        for j in range(ev.index - 1, max(0, ev.index - 10), -1):
            if ev.direction == 'bullish' and closes[j] < opens[j]:
                obs.append(OrderBlock('bullish', float(highs[j]), float(lows[j]), j))
                break
            elif ev.direction == 'bearish' and closes[j] > opens[j]:
                obs.append(OrderBlock('bearish', float(highs[j]), float(lows[j]), j))
                break
    return obs


def detect_displacement(df: pd.DataFrame, index: int, lookback: int = 10) -> bool:
    if index < lookback:
        return False
    body = abs(df['Close'].iloc[index] - df['Open'].iloc[index])
    past_bodies = abs(df['Close'].iloc[max(0, index - lookback):index] -
                      df['Open'].iloc[max(0, index - lookback):index])
    avg_body = past_bodies.mean()
    return bool(body > avg_body * 1.5) if avg_body > 0 else False

# ============================================================================
# VWAP 計算器（累積式）
# ============================================================================

class VWAPCalculator:

    @staticmethod
    def calculate(df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 5:
            last_price = float(df['Close'].iloc[-1]) if not df.empty else 0.0
            return {
                'vwap': last_price, 'upper_1': last_price, 'lower_1': last_price,
                'upper_2': last_price, 'lower_2': last_price,
                'series_vwap': pd.Series(dtype=float),
                'price_position': 'at',
            }

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cum_tp_vol = (typical_price * df['Volume']).cumsum()
        cum_vol = df['Volume'].cumsum().replace(0, np.nan)
        vwap_series = (cum_tp_vol / cum_vol).ffill().fillna(df['Close'])

        win = min(20, len(df))
        deviation = (typical_price - vwap_series).rolling(window=win).std().fillna(0)

        current_vwap = float(vwap_series.iloc[-1])
        current_dev = float(deviation.iloc[-1])
        current_price = float(df['Close'].iloc[-1])

        threshold = current_vwap * 0.001
        if current_price > current_vwap + threshold:
            price_position = 'above'
        elif current_price < current_vwap - threshold:
            price_position = 'below'
        else:
            price_position = 'at'

        return {
            'vwap': current_vwap,
            'upper_1': current_vwap + current_dev,
            'lower_1': current_vwap - current_dev,
            'upper_2': current_vwap + 2.0 * current_dev,
            'lower_2': current_vwap - 2.0 * current_dev,
            'series_vwap': vwap_series,
            'price_position': price_position,
        }

    @staticmethod
    def detect_bounce(df: pd.DataFrame, vwap_data: Dict) -> Dict:
        result = {'bounce_long': False, 'rejection_short': False,
                  'near_vwap': False, 'bounce_strength': 'none'}
        if len(df) < 5 or not vwap_data.get('vwap'):
            return result

        current_vwap = vwap_data['vwap']
        current_price = float(df['Close'].iloc[-1])
        prev_price = float(df['Close'].iloc[-2])

        near_threshold = current_vwap * 0.003
        result['near_vwap'] = abs(current_price - current_vwap) < near_threshold

        lookback = min(5, len(df) - 1)
        touch_threshold = current_vwap * 0.002
        recent_lows = df['Low'].iloc[-lookback:-1]
        recent_highs = df['High'].iloc[-lookback:-1]

        touched_from_above = any(low <= current_vwap + touch_threshold for low in recent_lows)
        touched_from_below = any(high >= current_vwap - touch_threshold for high in recent_highs)

        avg_vol = df['Volume'].tail(5).mean()
        vol_surge = float(df['Volume'].iloc[-1]) > avg_vol * 1.2 if avg_vol > 0 else False

        if touched_from_above and current_price > current_vwap and current_price > prev_price:
            result['bounce_long'] = True
            result['bounce_strength'] = 'strong' if vol_surge else 'weak'
        if touched_from_below and current_price < current_vwap and current_price < prev_price:
            result['rejection_short'] = True
            result['bounce_strength'] = 'strong' if vol_surge else 'weak'
        return result

    @staticmethod
    def get_score_contribution(df: pd.DataFrame, vwap_data: Dict, smc_bias: str) -> Tuple[int, List[str]]:
        score = 0
        signals: List[str] = []
        if not vwap_data or not vwap_data.get('vwap'):
            return 0, signals

        position = vwap_data.get('price_position', 'at')
        bounce = VWAPCalculator.detect_bounce(df, vwap_data)

        if smc_bias == 'bullish' and position == 'above':
            score += 10
            signals.append("📊 VWAP上方(+10)")
        elif smc_bias == 'bearish' and position == 'below':
            score += 10
            signals.append("📊 VWAP下方(+10)")
        elif position != 'at' and smc_bias != 'neutral':
            score -= 5
            signals.append("⚠️ VWAP逆向(-5)")

        if bounce['bounce_long'] and smc_bias == 'bullish':
            score += 15
            signals.append(f"🔄 VWAP Bounce({bounce['bounce_strength']}+15)")
        elif bounce['rejection_short'] and smc_bias == 'bearish':
            score += 15
            signals.append(f"🔄 VWAP Rejection({bounce['bounce_strength']}+15)")

        return score, signals

# ============================================================================
# ATR 計算
# ============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        atr_series = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
        val = atr_series.iloc[-1]
        if np.isnan(val) or val <= 0:
            tr = (df['High'] - df['Low']).tail(5).mean()
            return float(tr) if tr > 0 else float(df['Close'].iloc[-1] * 0.02)
        return float(val)
    except Exception:
        return float(df['Close'].iloc[-1] * 0.02)

# ============================================================================
# K 線取得（yfinance）
# ============================================================================

def get_kbars(symbol: str, interval: str = '5m', period: str = '5d') -> pd.DataFrame:
    """
    透過 yfinance 取得台股 K 線（加 .TW 後綴）

    Args:
        symbol: 純數字代碼（如 '2330'）
        interval: '5m' / '15m' / '1h' / '1d'
        period: '1d' / '5d' / '1mo'
    """
    ticker = f"{symbol}.TW"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            logger.warning(f"{ticker} yfinance 無資料")
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        logger.warning(f"{ticker} yfinance 失敗: {e}")
        return pd.DataFrame()


def get_latest_price_shioaji(symbol: str) -> float:
    """透過 Shioaji 取得即時報價"""
    api = get_shioaji_api()
    if api is None:
        return 0.0
    try:
        contract = api.Contracts.Stocks[symbol]
        if contract is None:
            logger.warning(f"{symbol} Shioaji 找不到合約")
            return 0.0
        snapshots = api.snapshots([contract])
        if snapshots:
            return float(snapshots[0].close)
        return 0.0
    except Exception as e:
        logger.debug(f"{symbol} Shioaji 報價失敗: {e}")
        return 0.0


def get_latest_price(symbol: str) -> float:
    """取得最新價格：優先 subscribe 快取 → Shioaji snapshot → yfinance"""
    # 2026-03-11 升級：改用 get_live_price（含 subscribe 快取）
    return get_live_price(symbol)

# ============================================================================
# 2026-03-11 升級：全域 Tick 報價字典（subscribe 事件驅動）
# ============================================================================

_tick_prices: Dict[str, float] = {}  # symbol → 最新成交價


def _on_tick_stk_v1(exchange, tick) -> None:
    """Shioaji Tick v1 callback — 更新全域報價字典"""
    try:
        _tick_prices[tick.code] = float(tick.close)
    except Exception:
        pass


def get_tick_price(symbol: str) -> float:
    """從 Tick 字典讀取最新價格（subscribe 推送），fallback 到 snapshots"""
    price = _tick_prices.get(symbol, 0.0)
    if price > 0:
        return price
    return get_latest_price(symbol)

# ============================================================================
# Kill Zone 判斷
# ============================================================================

def get_kill_zone_tw() -> Optional[str]:
    now_t = _tw_now().time()
    if CONFIG['KZ1_START'] <= now_t <= CONFIG['KZ1_END']:
        return 'kz1_open'
    elif CONFIG['KZ2_START'] <= now_t <= CONFIG['KZ2_END']:
        return 'kz2_close'
    return None

# ============================================================================
# SMC 評分（多時框）
# ============================================================================

def calculate_smc_score_tw(
    df_5m: pd.DataFrame,
    df_15m: Optional[pd.DataFrame] = None,
    df_1h: Optional[pd.DataFrame] = None,
    vwap_data: Optional[Dict] = None,
) -> Dict:
    """
    台股 SMC + VWAP 多層確認評分（0–100）

    1H 趨勢偏向        +15
    15M OB 確認         +20
    15M FVG 確認        +10
    5M CHoCH            +20
    5M BOS              +10
    Sweep→CHoCH         +15
    CHoCH後FVG          +10
    Kill Zone           +8~+15
    多時框方向一致      +10
    FVG 飽和度警告      -5（FVG > 6）
    多時框衝突          -15（1H vs 15M/5M 不同向）
    VWAP（最高+25）     +10/+15/-5
    """
    result: Dict = {
        'score': 0, 'bias': 'neutral', 'signals': [],
        'entry_zone': (0.0, 0.0), 'stop_price': 0.0, 'latest_event': '',
        'vwap_score': 0,
    }

    htf_bias = Bias.NEUTRAL
    bias_1h = Bias.NEUTRAL
    bias_15m = Bias.NEUTRAL
    bias_5m = Bias.NEUTRAL

    # 1H
    if df_1h is not None and len(df_1h) >= 15:
        try:
            swings_1h = find_swing_points(df_1h, lookback=3)
            events_1h = detect_structure(df_1h, swings_1h)
            if events_1h:
                latest_1h = events_1h[-1]
                if latest_1h.type in ('bos', 'choch'):
                    htf_bias = Bias.BULLISH if latest_1h.direction == 'bullish' else Bias.BEARISH
                    bias_1h = htf_bias
                    result['bias'] = htf_bias.value
                    result['score'] += 15
                    result['signals'].append(f"1H {latest_1h.type.upper()} {latest_1h.direction}")
        except Exception as e:
            logger.debug(f"1H SMC 失敗: {e}")

    # 15M
    if df_15m is not None and len(df_15m) >= 20:
        try:
            swings_15m = find_swing_points(df_15m, lookback=3)
            events_15m = detect_structure(df_15m, swings_15m)
            obs_15m = detect_order_blocks(df_15m, events_15m)
            fvgs_15m = check_fvg_mitigation(detect_fvg(df_15m), df_15m)

            if events_15m:
                latest_15m = events_15m[-1]
                if latest_15m.type in ('bos', 'choch'):
                    bias_15m = Bias.BULLISH if latest_15m.direction == 'bullish' else Bias.BEARISH

            if obs_15m:
                ob = obs_15m[-1]
                if not ob.mitigated:
                    result['entry_zone'] = (ob.low, ob.high)
                    result['score'] += 20
                    result['signals'].append(f"15M OB({ob.type})")

            active_fvgs = [f for f in fvgs_15m if not f.mitigated]
            if active_fvgs:
                # FVG 最小值檢驗：< 3 個拒絕進場
                if len(active_fvgs) < 3:
                    return {'score': -1, 'bias': result['bias'], 'signals': result['signals'] + [f"❌ FVG x{len(active_fvgs)} 不足，拒絕進場"],
                            'entry_zone': (0.0, 0.0), 'stop_price': 0.0, 'latest_event': '', 'vwap_score': 0}
                # FVG 飽和度檢查：≥7 時拒絕進場（2026-03-05 教訓）
                fvg_count = len(active_fvgs)
                if fvg_count >= CONFIG['FVG_SATURATION_THRESHOLD']:  # ≥7 時飽和
                    return {'score': -1, 'bias': result['bias'], 'signals': result['signals'] + [f"❌ FVG x{fvg_count} 飽和，拒絕進場"],
                            'entry_zone': None}
                
                # FVG 正常計分
                fvg_score = 10
                result['signals'].append(f"15M FVG x{fvg_count}")
                result['score'] += fvg_score
        except Exception as e:
            logger.debug(f"15M SMC 失敗: {e}")

    # 5M
    if len(df_5m) >= 20:
        try:
            swings_5m = find_swing_points(df_5m, lookback=3)
            events_5m = detect_structure(df_5m, swings_5m)
            fvgs_5m = check_fvg_mitigation(detect_fvg(df_5m), df_5m)

            if events_5m:
                latest_5m = events_5m[-1]
                if latest_5m.type in ('bos', 'choch'):
                    bias_5m = Bias.BULLISH if latest_5m.direction == 'bullish' else Bias.BEARISH
                result['latest_event'] = f"{latest_5m.type}_{latest_5m.direction}"

                if latest_5m.type == 'choch':
                    result['score'] += 20
                    result['signals'].append(f"5M CHoCH {latest_5m.direction}")
                elif latest_5m.type == 'bos':
                    result['score'] += 10
                    result['signals'].append(f"5M BOS {latest_5m.direction}")

                sweeps = [e for e in events_5m if e.type == 'sweep']
                chochs = [e for e in events_5m if e.type == 'choch']
                if sweeps and chochs:
                    if chochs[-1].index > sweeps[-1].index and chochs[-1].index - sweeps[-1].index <= 10:
                        result['score'] += 15
                        result['signals'].append("Sweep→CHoCH")

            active_fvgs_5m = [f for f in fvgs_5m if not f.mitigated]
            if active_fvgs_5m and events_5m:
                for fvg in active_fvgs_5m[-3:]:
                    for ev in events_5m[-5:]:
                        if ev.type == 'choch' and fvg.index > ev.index:
                            result['score'] += 10
                            result['signals'].append(f"CHoCH後FVG({fvg.type})")
                            if result['entry_zone'] == (0.0, 0.0):
                                result['entry_zone'] = (fvg.bottom, fvg.top)
                            break
        except Exception as e:
            logger.debug(f"5M SMC 失敗: {e}")

    # Kill Zone
    kz = get_kill_zone_tw()
    if kz == 'kz1_open':
        result['score'] += 15
        result['signals'].append("🟢 KZ1 開盤")
    elif kz == 'kz2_close':
        result['score'] += 8
        result['signals'].append("🟠 KZ2 尾盤")
    else:
        result['score'] -= 5
        result['signals'].append("⚠️ 非KZ時段")

    # 多時框方向一致 / 衝突檢測
    timeframes_with_bias = [
        (bias_1h, "1H"),
        (bias_15m, "15M"),
        (bias_5m, "5M"),
    ]
    active_biases = [(b, tf) for b, tf in timeframes_with_bias if b != Bias.NEUTRAL]
    
    if len(active_biases) >= 2:
        # 檢查是否所有時框方向一致
        bias_values = [b.value for b, _ in active_biases]
        if len(set(bias_values)) == 1:
            result['score'] += 10
            result['signals'].append("✓ 多時框方向一致")
        else:
            # 多時框衝突：1H vs 15M/5M 不同向 = 陷阱
            result['score'] -= 15
            conflicting_tfs = "/".join([tf for b, tf in active_biases])
            result['signals'].append(f"❌ 多時框衝突 ({conflicting_tfs})")
            logger.warning(f"多時框衝突警告: {conflicting_tfs} 方向不一致")

    # VWAP
    if vwap_data is not None:
        try:
            v_score, v_signals = VWAPCalculator.get_score_contribution(df_5m, vwap_data, result.get('bias', 'neutral'))
            result['score'] += v_score
            result['vwap_score'] = v_score
            result['signals'].extend(v_signals)
        except Exception as e:
            logger.debug(f"VWAP 評分失敗: {e}")

    result['score'] = max(0, min(100, result['score']))
    return result

# ============================================================================
# 風控系統
# ============================================================================

class DailyRiskManager:
    def __init__(self, initial_balance: float) -> None:
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_pnl: float = 0.0
        self.trade_count: int = 0
        self.consecutive_losses: int = 0
        self.is_locked: bool = False
        self.lock_reason: str = ""
        self.lock_time: Optional[float] = None
        self.max_daily_loss = initial_balance * CONFIG['DAILY_LOSS_LIMIT']
        logger.info(f"RiskManager: 餘額={initial_balance:,.0f}, 日損上限={self.max_daily_loss:,.0f}")

    def can_trade(self) -> Tuple[bool, str]:
        if self.is_locked and self.lock_time:
            elapsed = time.time() - self.lock_time
            if elapsed >= CONFIG['COOLDOWN_SEC']:
                self.is_locked = False
                self.consecutive_losses = 0
                self.lock_time = None
                self.lock_reason = ""
                logger.info("🔓 冷卻期結束，恢復交易")
        if self.is_locked:
            remaining = int(CONFIG['COOLDOWN_SEC'] - (time.time() - self.lock_time)) if self.lock_time else 0
            return False, f"{self.lock_reason}（剩餘 {remaining}s）"
        return True, "OK"

    def record_trade(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self.trade_count += 1
        self.current_balance += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self._check_limits()

    def _check_limits(self) -> None:
        if self.daily_pnl <= -self.max_daily_loss:
            self.is_locked = True
            self.lock_reason = f"日損上限 ({self.daily_pnl:,.0f}/{-self.max_daily_loss:,.0f})"
            self.lock_time = time.time()
            logger.warning(f"🔒 {self.lock_reason}")
        if self.consecutive_losses >= CONFIG['MAX_CONSECUTIVE_LOSSES']:
            self.is_locked = True
            self.lock_reason = f"連虧 {self.consecutive_losses} 筆"
            self.lock_time = time.time()
            logger.warning(f"🔒 {self.lock_reason}")

    def calculate_position_size(self, entry_price: float, stop_price: float,
                                win_rate: float = 0.5, smc_score: int = 75) -> int:
        """計算部位（整張＝1000股，不做零股）
        
        弱信號（< 80分）時，部位上限縮減為 1000 股；
        強信號（≥ 85分）時，允許 2000 股。
        """
        avg_win_pct = 0.03
        avg_loss_pct = 0.015
        R = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 2.0
        f_kelly = win_rate - (1 - win_rate) / R
        f_kelly = max(0.01, min(0.10, f_kelly))
        f_conservative = f_kelly * CONFIG['KELLY_FRACTION']

        risk_amount = self.current_balance * f_conservative
        risk_amount = min(risk_amount, self.current_balance * CONFIG['MAX_SINGLE_RISK_PCT'])

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)
        shares = (shares // 1000) * 1000  # 整張交易，無條件捨去到千股

        # SMC 評分影響部位上限
        if smc_score < 80:
            max_shares = 1000  # 弱信號時只買 1 張
        elif smc_score >= 85:
            max_shares = 2000  # 強信號時可買 2 張
        else:
            max_shares = int(self.current_balance * 0.10 / entry_price)
            max_shares = (max_shares // 1000) * 1000
        
        shares = min(shares, max_shares)
        return shares  # 0 = 買不起整張，跳過該標的

# ============================================================================
# 持倉持久化
# ============================================================================

def save_positions(positions: Dict) -> None:
    try:
        with open(positions_path(), 'w', encoding='utf-8') as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"儲存持倉失敗: {e}")

def load_positions() -> Dict:
    p = positions_path()
    if not p.exists():
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

# ============================================================================
# 交易日誌
# ============================================================================

@dataclass
class TradeRecord:
    symbol: str
    side: str
    qty: int
    entry_price: float
    exit_price: float
    stop_price: float
    take_profit: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str
    smc_score: int
    smc_signals: List[str]
    kill_zone: str

def save_trade_log(record: TradeRecord) -> None:
    log_file = trade_log_path()
    existing: List[Dict] = []
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(asdict(record))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

# ============================================================================
# 盤前篩選
# ============================================================================

def premarket_screener(watchlist: List[str] = None) -> List[str]:
    """篩選股票池：檢查價格、成交量"""
    stocks = watchlist or DEFAULT_WATCHLIST
    logger.info(f"=== 盤前篩選: {len(stocks)} 支 ===")
    result = []
    for sym in stocks:
        try:
            df = get_kbars(sym, interval='1d', period='5d')
            if df.empty or len(df) < 2:
                continue
            price = float(df['Close'].iloc[-1])
            vol = float(df['Volume'].iloc[-1])
            avg_vol = float(df['Volume'].tail(5).mean())

            if price < 10 or vol < 500:
                continue
            if price > CONFIG['MAX_STOCK_PRICE']:
                logger.debug(f"  {sym}: ${price:.2f} 超過整張上限 ${CONFIG['MAX_STOCK_PRICE']:.0f}，跳過")
                continue

            change_pct = (price - float(df['Close'].iloc[-2])) / float(df['Close'].iloc[-2]) * 100

            result.append(sym)
            logger.info(f"  {sym}: ${price:.2f} vol={vol:,.0f} chg={change_pct:+.2f}% ✓")
        except Exception as e:
            logger.debug(f"  {sym} 篩選失敗: {e}")

    logger.info(f"=== 篩選完成: {len(result)} 支通過 ===")
    return result if result else stocks[:10]

# ============================================================================
# 漲跌停偵測
# ============================================================================

def check_limit(symbol: str, current_price: float, prev_close: float) -> Optional[str]:
    """偵測漲跌停"""
    if prev_close <= 0:
        return None
    change_pct = (current_price - prev_close) / prev_close
    if change_pct >= CONFIG['LIMIT_PCT'] - 0.001:
        return 'limit_up'
    elif change_pct <= -(CONFIG['LIMIT_PCT'] - 0.001):
        return 'limit_down'
    return None

# ============================================================================
# 主交易引擎
# ============================================================================

class TWDaytradeBot:
    def __init__(self) -> None:
        self.risk_mgr = DailyRiskManager(CONFIG['INITIAL_BALANCE'])
        self.positions: Dict[str, Dict] = load_positions()
        self.watchlist: List[str] = []
        # 2026-03-11 升級：Shioaji API 實例 + 訂閱追蹤 + 流量監控
        self._sj_api = None
        self._last_usage_check: float = 0.0
        self._subscribed: set = set()

        if self.positions:
            logger.info(f"從檔案恢復 {len(self.positions)} 個持倉: {list(self.positions.keys())}")

    @staticmethod
    def _is_twse_trading_day() -> bool:
        """檢查今天是否為台股交易日（排除週末 + 國定假日）"""
        today = _tw_now().date()
        # 週末
        if today.weekday() >= 5:
            return False
        # 2026 台股國定假日（含補假、彈性放假）
        holidays_2026 = {
            date(2026, 1, 1),   # 元旦
            date(2026, 1, 2),   # 元旦彈性放假
            date(2026, 2, 14),  # 除夕前（調整放假）
            date(2026, 2, 16),  # 除夕
            date(2026, 2, 17),  # 春節
            date(2026, 2, 18),  # 春節
            date(2026, 2, 19),  # 春節
            date(2026, 2, 20),  # 春節（彈性放假）
            date(2026, 2, 27),  # 和平紀念日（彈性放假）
            date(2026, 2, 28),  # 和平紀念日（週六，但列入）
            date(2026, 4, 3),   # 兒童節（彈性放假）
            date(2026, 4, 4),   # 清明節
            date(2026, 4, 5),   # 兒童節
            date(2026, 5, 1),   # 勞動節
            date(2026, 5, 25),  # 端午節
            date(2026, 10, 5),  # 中秋節
            date(2026, 10, 10), # 國慶日
        }
        return today not in holidays_2026

    def run(self) -> None:
        if not self._is_twse_trading_day():
            logger.info(f"📅 今日 {_tw_now().date()} 非台股交易日，Bot 不啟動")
            _tg_notify(f"📅 台股今日休市（{_tw_now().date()}），Bot 不啟動")
            return

        logger.info(f"🚀 TW DayTrade Bot v{__version__} 啟動")
        logger.info(f"交易時段: 09:00–13:30")
        logger.info(f"KZ1 開盤: 09:00–10:00, KZ2 尾盤: 12:30–13:20")
        logger.info(f"強制平倉: 13:20, 不開新倉: 13:00 之後")

        # 2026-03-06 修復：啟動時立即登入 Shioaji，不再延遲初始化
        # 避免進場後才觸發登入，導致監控迴圈卡死或 crash
        sj_api = get_shioaji_api()
        if sj_api is None:
            logger.critical("❌ Shioaji 登入失敗，Bot 無法啟動")
            _tg_notify("❌ 台股 Bot 啟動失敗：Shioaji 登入失敗")
            return
        logger.info("✅ Shioaji 連線確認完成，開始盤前篩選")

        self.watchlist = premarket_screener()
        logger.info(f"監控清單: {self.watchlist}")

        # 2026-03-11 升級：訂閱即時報價（事件驅動）
        subscribe_quotes(self.watchlist)

        # 2026-03-11 升級：流量計時器
        self._last_usage_check = time.time()

        try:
            while True:
                self._loop()
        except KeyboardInterrupt:
            logger.info("Bot 停止（使用者中斷）")
            self._close_all("手動停止")
        except Exception as e:
            logger.critical(f"主迴圈異常: {e}\n{traceback.format_exc()}")
            self._close_all("緊急平倉")
        finally:
            # 2026-03-11 升級：清理訂閱
            unsubscribe_all()

    def _loop(self) -> None:
        now_t = _tw_now().time()

        # 非交易時段
        if now_t > CONFIG['MARKET_CLOSE']:
            logger.info("📴 收盤，Bot 結束運行")
            if self.positions:
                self._close_all("收盤強制平倉")
            sys.exit(0)

        if now_t < CONFIG['MARKET_OPEN']:
            logger.debug("非交易時段（等待開盤）")
            time.sleep(60)
            return

        # 強制平倉
        if now_t >= CONFIG['FORCE_CLOSE']:
            if self.positions:
                logger.info("⏰ 13:20 強制平倉")
                self._close_all("強制平倉 13:20")
            time.sleep(60)
            return

        # 風控
        can, reason = self.risk_mgr.can_trade()
        if not can:
            logger.info(f"🔒 {reason}")
            if self.positions:
                self._close_all(f"風控: {reason}")
            time.sleep(60)
            return

        # 有持倉：監控
        if self.positions:
            self._monitor_positions()
            time.sleep(CONFIG['MONITOR_INTERVAL_SEC'])
            return

        # 2026-03-11 升級：每 5 分鐘檢查 API 流量
        if hasattr(self, '_last_usage_check') and time.time() - self._last_usage_check > 300:
            shioaji_check_usage()
            self._last_usage_check = time.time()

        # Kill Zone 掃描
        kz = get_kill_zone_tw()
        if kz and now_t < CONFIG['NO_NEW_TRADE_AFTER'] and len(self.positions) < CONFIG['MAX_POSITIONS']:
            self._scan_entries(kz)
        time.sleep(CONFIG['SCAN_INTERVAL_SEC'])

    def _scan_entries(self, kz: str) -> None:
        logger.info(f"=== 掃描入場 [{kz}] — {len(self.watchlist)} 支 ===")
        for symbol in self.watchlist:
            if symbol in self.positions:
                continue
            if len(self.positions) >= CONFIG['MAX_POSITIONS']:
                break
            try:
                self._analyze_and_enter(symbol, kz)
            except Exception as e:
                logger.warning(f"{symbol} 分析失敗: {e}")

    def _analyze_and_enter(self, symbol: str, kz: str) -> None:
        df_5m = get_kbars(symbol, '5m', '5d')
        df_15m = get_kbars(symbol, '15m', '5d')
        df_1h = get_kbars(symbol, '1h', '1mo')

        if df_5m.empty or len(df_5m) < 20:
            return

        current_price = float(df_5m['Close'].iloc[-1])
        if current_price <= 0:
            return

        # ATR: 1h 優先，fallback 日線 × 0.3
        if not df_1h.empty and len(df_1h) >= 14:
            atr = calculate_atr(df_1h)
        else:
            df_1d = get_kbars(symbol, '1d', '1mo')
            if not df_1d.empty and len(df_1d) >= 14:
                atr = calculate_atr(df_1d) * 0.3
            else:
                atr = current_price * 0.02

        # 漲跌停檢查
        df_1d_check = get_kbars(symbol, '1d', '5d')
        if not df_1d_check.empty and len(df_1d_check) >= 2:
            prev_close = float(df_1d_check['Close'].iloc[-2])
            limit_status = check_limit(symbol, current_price, prev_close)
            if limit_status == 'limit_down':
                logger.info(f"{symbol} 跌停，跳過")
                return
            if limit_status == 'limit_up':
                logger.info(f"{symbol} 漲停，跳過（追高風險）")
                return

        # VWAP
        vwap_data = VWAPCalculator.calculate(df_5m)

        # SMC 評分
        smc = calculate_smc_score_tw(
            df_5m,
            df_15m if not df_15m.empty else None,
            df_1h if not df_1h.empty else None,
            vwap_data=vwap_data,
        )
        score = smc['score']
        bias = smc['bias']
        signals = smc['signals']

        logger.debug(
            f"{symbol}: ${current_price:.2f} ATR={atr:.2f} "
            f"VWAP=${vwap_data['vwap']:.2f}({vwap_data['price_position']}) "
            f"SMC={score} bias={bias}"
        )

        # 只做多
        if bias != 'bullish':
            return
        
        # 【新增】多時框衝突檢查：即使高分也要降權
        # 2026-03-04 檢討：多時框衝突是進場失敗的根本原因
        has_timeframe_conflict = any('多時框衝突' in sig for sig in signals)
        if has_timeframe_conflict and kz == 'kz1_open':
            # KZ1 開盤期多時框衝突特別危險（假突破比例高）
            # 降權策略：降低 30 分 或 直接拒絕進場
            score = max(0, score - 30)
            logger.warning(f"{symbol} KZ1 檢出多時框衝突，評分降權至 {score}")
        
        # 根據 Kill Zone 調整評分門檻
        if kz == 'kz1_open':
            min_score = CONFIG['KZ1_MIN_SCORE']  # 開盤期評分門檻
        elif kz == 'kz2_close':
            min_score = 85  # KZ2 尾盤提升到 85（相比一般的 75）
        else:
            min_score = CONFIG['MIN_SCORE']  # 一般時段
        
        if score < min_score:
            logger.debug(f"{symbol}: SMC={score} 未達{kz}的門檻({min_score})，跳過")
            return

        # VWAP 距離檢查：進場點不超過 VWAP 距離 1.5 倍 ATR（防止假突破）
        vwap_price = vwap_data['vwap']
        vwap_distance = abs(current_price - vwap_price)
        max_vwap_distance = CONFIG['ATR_STOP_MULT'] * atr  # 1.5 倍 ATR
        if vwap_distance > max_vwap_distance:
            logger.debug(
                f"{symbol}: 進場距離 VWAP 過遠 ({vwap_distance:.2f} > {max_vwap_distance:.2f})，跳過"
            )
            return

        # 停損停利
        stop_price = current_price - CONFIG['ATR_STOP_MULT'] * atr
        take_profit = current_price + CONFIG['ATR_TP_MULT'] * atr

        # 部位（考慮 SMC 評分強度）
        qty = self.risk_mgr.calculate_position_size(current_price, stop_price, smc_score=score)
        if qty <= 0:
            return

        # 2026-03-11 升級：透過 Shioaji 下單（模擬模式）
        logger.info(
            f"📈 進場信號: {symbol} @ {current_price:.2f} "
            f"qty={qty} stop={stop_price:.2f} tp={take_profit:.2f} "
            f"SMC={score} [{', '.join(signals[:3])}]"
        )

        order_result = shioaji_place_order(symbol, 'buy', qty)
        if order_result is None:
            logger.warning(f"{symbol} 下單失敗，放棄進場")
            return

        filled_price = order_result['filled_price']
        if filled_price <= 0:
            filled_price = current_price  # fallback
        actual_qty = order_result['qty_shares']

        # 用實際成交價重算停損停利
        stop_price = filled_price - CONFIG['ATR_STOP_MULT'] * atr
        take_profit = filled_price + CONFIG['ATR_TP_MULT'] * atr

        self.positions[symbol] = {
            'qty': actual_qty,
            'entry_price': filled_price,
            'stop_price': stop_price,
            'take_profit': take_profit,
            'atr': atr,
            'entry_time': _tw_now().isoformat(),
            'smc_score': score,
            'smc_signals': signals,
            'kill_zone': kz,
            'highest_price': filled_price,
            'order_id': order_result.get('order_id', ''),
        }
        _tg_notify(
            f"📈 <b>台股買入</b> [{kz}]\n"
            f"<b>{symbol}</b> {actual_qty}股 @ ${filled_price:.2f}\n"
            f"停損: ${stop_price:.2f} | 停利: ${take_profit:.2f}\n"
            f"SMC: {score} | {', '.join(signals[:3])}"
        )
        save_positions(self.positions)

    def _monitor_positions(self) -> None:
        to_close: List[Tuple[str, str]] = []
        current_time = _tw_now().time()
        
        for symbol, pos in list(self.positions.items()):
            try:
                current_price = get_latest_price(symbol)
                if current_price <= 0:
                    continue

                # 移動停損
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                    trailing_stop = current_price - CONFIG['ATR_STOP_MULT'] * pos['atr']
                    if trailing_stop > pos['stop_price']:
                        pos['stop_price'] = trailing_stop
                        logger.debug(f"{symbol} 移動停損: {trailing_stop:.2f}")

                # 進場冷卻
                entry_time = datetime.fromisoformat(pos['entry_time'])
                elapsed = (_tw_now() - entry_time).total_seconds()
                if elapsed < CONFIG['ENTRY_COOLDOWN_SEC']:
                    continue

                exit_reason: Optional[str] = None

                if current_price <= pos['stop_price']:
                    exit_reason = 'stop_loss'
                elif current_price >= pos['take_profit']:
                    exit_reason = 'take_profit'

                # 強制平倉（優先級最高，必須無條件執行）
                if current_time >= CONFIG['FORCE_CLOSE']:
                    exit_reason = 'force_close'

                # 跌停偵測
                if not exit_reason:  # 未被強制平倉才檢查跌停
                    df_1d = get_kbars(symbol, '1d', '5d')
                    if not df_1d.empty and len(df_1d) >= 2:
                        prev_close = float(df_1d['Close'].iloc[-2])
                        if check_limit(symbol, current_price, prev_close) == 'limit_down':
                            exit_reason = 'limit_down'
                            logger.warning(f"{symbol} 跌停！以跌停價平倉")

                if exit_reason:
                    to_close.append((symbol, exit_reason))
            except Exception as e:
                logger.error(f"{symbol} 監控異常: {e}")

        for symbol, reason in to_close:
            self._exit_position(symbol, reason)

    def _exit_position(self, symbol: str, reason: str) -> None:
        pos = self.positions.get(symbol)
        if not pos:
            return

        # 2026-03-11 升級：透過 Shioaji 下賣出單
        sell_result = shioaji_place_order(symbol, 'sell', pos['qty'])
        if sell_result and sell_result['filled_price'] > 0:
            current_price = sell_result['filled_price']
        else:
            # 賣單失敗時 fallback：用即時報價計算
            current_price = get_latest_price(symbol)
            if current_price <= 0:
                current_price = pos['entry_price']
            if sell_result is None:
                logger.warning(f"{symbol} 賣出下單失敗，以即時價 ${current_price:.2f} 記帳")

        pnl = (current_price - pos['entry_price']) * pos['qty']
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']

        logger.info(
            f"{'✅' if pnl >= 0 else '❌'} 出場: {symbol} "
            f"入={pos['entry_price']:.2f} 出={current_price:.2f} "
            f"qty={pos['qty']} PnL={pnl:+,.0f} ({pnl_pct:+.2%}) 原因={reason}"
        )
        emoji = '💰' if pnl >= 0 else '🛑'
        reason_zh = {'stop_loss': '停損', 'take_profit': '停利', 'force_close': '強制平倉', 'trailing': '移動停損'}.get(reason, reason)
        daily_pnl = self.risk_mgr.daily_pnl + pnl
        _tg_notify(
            f"{emoji} <b>台股出場</b> — {reason_zh}\n"
            f"<b>{symbol}</b> {pos['qty']}股\n"
            f"入場: ${pos['entry_price']:.2f} → 出場: ${current_price:.2f}\n"
            f"PnL: <b>${pnl:+,.0f} ({pnl_pct:+.2%})</b>\n"
            f"日累計: ${daily_pnl:+,.0f}"
        )

        self.risk_mgr.record_trade(pnl)

        record = TradeRecord(
            symbol=symbol, side='buy', qty=pos['qty'],
            entry_price=pos['entry_price'], exit_price=current_price,
            stop_price=pos['stop_price'], take_profit=pos['take_profit'],
            pnl=pnl, pnl_pct=pnl_pct,
            entry_time=pos['entry_time'], exit_time=_tw_now().isoformat(),
            exit_reason=reason, smc_score=pos['smc_score'],
            smc_signals=pos['smc_signals'], kill_zone=pos['kill_zone'],
        )
        save_trade_log(record)
        del self.positions[symbol]
        save_positions(self.positions)

    def _close_all(self, reason: str) -> None:
        logger.info(f"=== 全部平倉: {reason} ===")
        for symbol in list(self.positions.keys()):
            try:
                self._exit_position(symbol, reason)
            except Exception as e:
                logger.error(f"{symbol} 平倉失敗: {e}")

# ============================================================================
# CLI
# ============================================================================

def cmd_screener() -> None:
    print("\n🔍 盤前篩選...")
    result = premarket_screener()
    print(f"候選股（{len(result)} 支）: {result}\n")

def cmd_status() -> None:
    print(f"\n📊 TW DayTrade Bot v{__version__}")
    print(f"  模擬帳戶餘額: {CONFIG['INITIAL_BALANCE']:,.0f}")
    positions = load_positions()
    if positions:
        print(f"  當前持倉: {len(positions)} 支")
        for sym, pos in positions.items():
            price = get_latest_price(sym)
            entry = pos['entry_price']
            pnl_pct = (price - entry) / entry * 100 if price > 0 and entry > 0 else 0
            print(f"    {sym}: 入={entry:.2f} 現={price:.2f} ({pnl_pct:+.2f}%) qty={pos['qty']}")
    else:
        print("  當前無持倉")
    now_t = _tw_now().time()
    kz = get_kill_zone_tw()
    in_market = CONFIG['MARKET_OPEN'] <= now_t <= CONFIG['MARKET_CLOSE']
    print(f"  交易時段: {'✅ 開盤中' if in_market else '❌ 休市'}")
    print(f"  Kill Zone: {kz or '非KZ'}")
    print()

def cmd_log() -> None:
    log_file = trade_log_path()
    if not log_file.exists():
        print("今日尚無交易記錄")
        return
    with open(log_file, 'r', encoding='utf-8') as f:
        trades = json.load(f)
    print(f"\n📋 今日交易（{len(trades)} 筆）")
    total_pnl = 0.0
    for t in trades:
        pnl = t.get('pnl', 0)
        total_pnl += pnl
        print(
            f"  {t['symbol']:6s} qty={t['qty']:>5} "
            f"入={t['entry_price']:.2f} 出={t['exit_price']:.2f} "
            f"PnL={pnl:+,.0f} ({t['pnl_pct']:+.2%}) {t['exit_reason']}"
        )
    print(f"\n  日累計: {total_pnl:+,.0f}\n")

def print_help() -> None:
    print(f"""
TW DayTrade Bot v{__version__} — 台股當沖模擬交易（Shioaji + SMC + VWAP）

用法:
  python3 tw-daytrade-bot.py [命令]

命令:
  run        啟動主交易循環（模擬）
  screener   盤前篩選
  status     帳戶狀態 + 持倉
  log        今日交易日誌
  help       說明

交易時段:
  09:00–13:30  交易時間
  KZ1 09:00–10:00  開盤 Kill Zone
  KZ2 12:30–13:20  尾盤 Kill Zone
  13:00 之後不開新倉
  13:20 強制平倉

評分系統（SMC + VWAP，門檻 ≥ {CONFIG['MIN_SCORE']}）:
  多時框 BOS/CHoCH（1H + 15M + 5M）
  OB / FVG 確認
  VWAP 位置 + Bounce
  Kill Zone 加分

風控:
  日損上限:   {CONFIG['DAILY_LOSS_LIMIT']*100:.1f}%
  連虧鎖倉:  {CONFIG['MAX_CONSECUTIVE_LOSSES']} 筆 + {CONFIG['COOLDOWN_SEC']//60} 分鐘冷卻
  Kelly 係數: {CONFIG['KELLY_FRACTION']}
  ATR 停損:   ×{CONFIG['ATR_STOP_MULT']}
  ATR 停利:   ×{CONFIG['ATR_TP_MULT']}
  交易單位:   整張（1000 股），買不起整張則跳過

⚠️ 模擬交易專用（simulation=True）
""")

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or args[0] in ('help', '--help', '-h'):
        print_help()
    elif args[0] == 'run':
        bot = TWDaytradeBot()
        bot.run()
    elif args[0] == 'screener':
        cmd_screener()
    elif args[0] == 'status':
        cmd_status()
    elif args[0] == 'log':
        cmd_log()
    else:
        print(f"未知命令: {args[0]}")
        print_help()
