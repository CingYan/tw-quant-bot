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

v2 重點修正（來自 trading-lessons.md）：
  - 多時框衝突 → HARD REJECT（非扣分）
  - FVG 飽和度 ≥7 → HARD REJECT
  - 連虧 3 筆 → 今日停止交易（非 30 分鐘後自動解鎖）
  - 交易記錄進場時立即寫入磁碟
  - Watchdog 主迴圈：永不永久崩潰
  - Heartbeat 每 30 秒寫入 /tmp/tw-bot-heartbeat
  - SIGTERM/SIGINT 優雅關閉
  - v2.1.3: KZ 內 weak VWAP bounce/rejection 一律硬拒絕
  - v2.1.4: KZ1 開盤寬限後必須有 strong VWAP interaction
  - v2.1.5: 移動停損出場不再誤算為 stop-loss ban
  - v2.1.6: 空單進場前檢查 Shioaji 合約可放空性，並避免下單錯誤 log 洩漏 payload
  - v2.1.7: 券商拒單 log 保留安全診斷欄位（action/lots/order_id/status/order_quantity）
  - v2.1.8: entry 下單失敗時保留候選上下文（direction/score/price/stop/tp/qty）
  - v2.1.9: 同檔當日連續券商拒單達上限後停止重送 entry order
  - v2.1.10: 空單券源未知時 fail closed，避免送出高機率被券商拒絕的 sell entry
  - v2.1.11: 持倉記錄 TSL 啟動狀態，避免 trailing-adjusted loss 被誤標為 initial SL
  - v2.1.12: Shioaji 分批成交用加權均價與實際成交股數記錄，避免 PnL 漏算
  - v2.1.13: opposing 15M OB 後同檔同方向冷卻，避免剛被反向供需區擋下又立刻追同向
  - v2.1.14: KZ1 同方向 initial SL 後冷卻，避免開盤假方向連續追單
  - v2.1.15: 重啟後從當日交易檔重建風控狀態（日 PnL/連虧/停損 ban/冷卻），
             避免盤中重啟或 watchdog 拉起時洗掉當日風控
  - v2.1.16: 空單券源餘額不可解析時 fail closed，避免模擬券商拒單噪音
  - v2.1.17: 券商拒單時補 contract/order 非敏感診斷欄位，追 short tradability root cause
  - v2.1.18: KZ1 全時段都必須有 strong VWAP interaction，不再給 09:00-09:05 位置-only 寬限
  - v2.1.19: 出場單被券商拒絕時保留持倉，不再誤記為已平倉
  - v2.1.20: 明確區分多空訂單意圖，移除偽成交 fallback，未知成交狀態隔離，
             空單券源改為數量檢查並 fail closed
  - v2.1.21: 未知成交狀態隔離持久化，避免盤中重啟後重送可能已成交的委託
  - v2.1.22: 收盤/關閉通知只在所有持倉確實平倉後送出；失敗持倉改發人工處理告警
  - v2.1.23: 券商登入例外改記錄安全摘要，避免上游 payload 洩漏敏感資訊
  - v2.1.24: 券商登入逾時後限頻重試並補回即時報價訂閱，避免整盤退回延遲資料

⚠️ 模擬交易專用 — simulation=True，禁止真實下單
"""

__version__ = "2.1.24"

import sys
import os
import json
import signal
import logging
import time
import traceback
from datetime import datetime, date, time as dt_time, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Path setup: allow importing v2.core
# ============================================================================
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'CAPITAL': 1_000_000,
    'MAX_DAILY_LOSS_PCT': 0.015,
    'MAX_POSITIONS': 3,
    'KELLY_FRACTION': 0.3,
    'MAX_SINGLE_RISK_PCT': 0.005,
    'ATR_STOP_MULT': 2.5,
    'ATR_TP_MULT': 3.5,
    'RE_ENTRY_COOLDOWN_SEC': 600,
    'ENTRY_COOLDOWN_SEC': 120,
    'OPPOSING_OB_COOLDOWN_SEC': 600,
    'KZ1_SIDE_STOP_COOLDOWN_SEC': 1800,
    'MIN_SCORE': 70,
    'KZ1_MIN_SCORE': 98,  # 2026-03-20: KZ1時段weak rejection失效，提高門檻至98
    'KZ2_MIN_SCORE': 98,  # 2026-03-19: KZ2時段勝率低，提高門檻至98
    'FVG_SATURATION_THRESHOLD': 7,
    'CONSECUTIVE_LOSS_LIMIT': 3,
    'HEARTBEAT_INTERVAL_SEC': 30,
    'NOTIFY_CHAT_ID': os.environ.get('DAYTRADE_NOTIFY_CHAT_ID', ''),
    'SCAN_INTERVAL_SEC': 30,
    'MONITOR_INTERVAL_SEC': 20,
    # A failed startup login must not leave the bot on delayed data for the
    # rest of the session.  Retry quote connectivity, but rate-limit broker
    # login attempts so an outage cannot stall every scan.
    'BROKER_RETRY_INTERVAL_SEC': 60,
    'ORDER_FAIL_DAILY_LIMIT': 2,
    # TW market hours
    'MARKET_OPEN': dt_time(9, 0),
    'MARKET_CLOSE': dt_time(13, 30),
    'KZ1_START': dt_time(9, 0),
    'KZ1_END': dt_time(10, 0),
    'KZ2_START': dt_time(12, 30),
    'KZ2_END': dt_time(13, 20),
    'NO_NEW_TRADE_AFTER': dt_time(12, 35),
    'FORCE_CLOSE': dt_time(13, 0),
    'OPENING_GRACE_END': dt_time(9, 5),   # first 5 min: score >= 95
    'INITIAL_BALANCE': 1_000_000.0,
    'MAX_STOCK_PRICE': 100.0,
}

# Default watchlist
DEFAULT_WATCHLIST = [
    '2303', '2881', '2882', '2891', '2886', '2884', '6505', '1301',
    '1303', '2002', '3481', '2409', '2603', '2609', '2615', '5880',
    '2890', '2883', '1326',
]

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(os.environ.get('DAYTRADE_BASE_DIR', Path(__file__).resolve().parent.parent))
CRED_FILE = Path(os.environ.get('DAYTRADE_CRED_FILE',
                                str(BASE_DIR / 'credentials' / 'credentials.json')))
DATA_DIR = BASE_DIR / 'data' / 'tw-daytrade'
DATA_DIR.mkdir(parents=True, exist_ok=True)
HEARTBEAT_FILE = Path('/tmp/tw-bot-heartbeat')

TW_TZ = timezone(timedelta(hours=8))


def _tw_now() -> datetime:
    return datetime.now(TW_TZ)


def _today_str() -> str:
    return _tw_now().strftime('%Y-%m-%d')


def bot_log_path() -> Path:
    return DATA_DIR / f'bot-{_today_str()}.log'


def unknown_order_quarantine_path() -> Path:
    """Per-trading-day fail-closed marker for unresolved broker orders."""
    return DATA_DIR / f'unknown-orders-{_today_str()}.json'


# ============================================================================
# Logging
# ============================================================================
def setup_logging() -> logging.Logger:
    lg = logging.getLogger('tw_daytrade_bot')
    lg.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(bot_log_path(), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    lg.addHandler(ch)
    return lg


logger = setup_logging()

# ============================================================================
# Credentials
# ============================================================================
def load_credentials() -> Dict:
    try:
        return json.loads(CRED_FILE.read_text(encoding='utf-8'))
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
# Lazy imports (installed by uv at runtime)
# ============================================================================
_pd = None
_np = None
_yf = None
_ta = None


def _ensure_deps():
    global _pd, _np, _yf, _ta
    if _pd is not None:
        return
    import pandas as __pd
    import numpy as __np
    import yfinance as __yf
    import ta as __ta
    _pd = __pd
    _np = __np
    _yf = __yf
    _ta = __ta


# ============================================================================
# Core module imports
# ============================================================================
from v2.core.smc_engine import SMCEngine, VWAPCalculator, calculate_atr
from v2.core.risk_manager import RiskManager
from v2.core.notification import TelegramNotifier
from v2.core.trade_recorder import TradeRecorder

# ============================================================================
# Shioaji global instance
# ============================================================================
_shioaji_api = None
_next_shioaji_login_retry_ts = 0.0
_live_prices: Dict[str, float] = {}
_live_prices_ts: Dict[str, float] = {}


def _safe_broker_error_summary(error: Exception) -> str:
    """Return an exception class only; broker messages can embed auth payloads."""
    return type(error).__name__ or 'BrokerError'
_subscribed_symbols: set = set()


def get_shioaji_api():
    global _shioaji_api, _next_shioaji_login_retry_ts
    if _shioaji_api is not None:
        return _shioaji_api
    if time.time() < _next_shioaji_login_retry_ts:
        return None
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
        _next_shioaji_login_retry_ts = 0.0
        return api
    except Exception as e:
        logger.error("Shioaji 登入失敗: %s", _safe_broker_error_summary(e))
        _next_shioaji_login_retry_ts = time.time() + CONFIG['BROKER_RETRY_INTERVAL_SEC']
        return None


def _on_tick_stk_v1(exchange, tick):
    code = getattr(tick, 'code', '')
    price = getattr(tick, 'close', 0.0)
    if code and price > 0:
        _live_prices[code] = float(price)
        _live_prices_ts[code] = time.time()


def subscribe_quotes(symbols: List[str]) -> None:
    api = get_shioaji_api()
    if api is None:
        return
    try:
        import shioaji as sj
        api.quote.set_on_tick_stk_v1_callback(_on_tick_stk_v1)
        for sym in symbols:
            if sym in _subscribed_symbols:
                continue
            try:
                contract = api.Contracts.Stocks.get(sym) or api.Contracts.Stocks[sym]
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
    except Exception as e:
        logger.warning(f"subscribe_quotes 失敗: {e}")


def unsubscribe_all() -> None:
    api = get_shioaji_api()
    if api is None:
        return
    try:
        import shioaji as sj
        for sym in list(_subscribed_symbols):
            try:
                contract = api.Contracts.Stocks.get(sym) or api.Contracts.Stocks[sym]
                if contract:
                    api.quote.unsubscribe(
                        contract,
                        quote_type=sj.constant.QuoteType.Tick,
                        version=sj.constant.QuoteVersion.v1,
                    )
            except Exception:
                pass
        _subscribed_symbols.clear()
    except Exception:
        pass


def get_live_price(symbol: str) -> float:
    # 1. subscribe cache (5s valid)
    if symbol in _live_prices:
        age = time.time() - _live_prices_ts.get(symbol, 0)
        if age < 5.0:
            return _live_prices[symbol]
    # 2. Shioaji snapshot
    api = get_shioaji_api()
    if api:
        try:
            contract = api.Contracts.Stocks[symbol]
            snaps = api.snapshots([contract])
            if snaps:
                price = float(snaps[0].close)
                _live_prices[symbol] = price
                _live_prices_ts[symbol] = time.time()
                return price
        except Exception:
            pass
    # 3. yfinance fallback
    try:
        _ensure_deps()
        df = _yf.download(f"{symbol}.TW", period='1d', interval='5m', progress=False)
        if not df.empty:
            if isinstance(df.columns, _pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return float(df['Close'].iloc[-1])
    except Exception:
        pass
    return 0.0


def _float_attr(obj, names: List[str]) -> float:
    """Read the first numeric attribute/key from Shioaji objects or dicts."""
    for name in names:
        value = None
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        try:
            if value is not None and value != '':
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _extract_fill_summary(trade, requested_lots: int) -> Tuple[float, int]:
    """Return weighted-average fill price and filled lots from Shioaji status/deals."""
    status = getattr(trade, 'status', None)
    deals = []
    for source in (trade, status):
        source_deals = getattr(source, 'deals', None) if source is not None else None
        if source_deals:
            deals = list(source_deals)
            break

    total_lots = 0.0
    total_value = 0.0
    seen = set()
    for deal in deals:
        price = _float_attr(deal, ['price', 'deal_price', 'modified_price'])
        qty = _float_attr(deal, ['quantity', 'qty', 'deal_quantity', 'volume'])
        deal_key = (
            getattr(deal, 'seq', None) if not isinstance(deal, dict) else deal.get('seq'),
            getattr(deal, 'ts', None) if not isinstance(deal, dict) else deal.get('ts'),
            price,
            qty,
        )
        if deal_key in seen:
            continue
        seen.add(deal_key)

        if price <= 0:
            continue
        if qty <= 0:
            qty = 1.0
        total_lots += qty
        total_value += price * qty

    if total_lots > 0 and total_value > 0:
        avg_price = total_value / total_lots
        if total_lots > requested_lots and total_lots % 1000 == 0:
            total_lots = total_lots / 1000
        return avg_price, int(total_lots)

    status_price = _float_attr(status, ['deal_price', 'modified_price'])
    # `order_quantity` is the requested quantity, not proof of a fill.  Never
    # use it as a fill fallback: an IOC can be accepted and then expire with
    # zero deals.
    status_qty = _float_attr(status, ['deal_quantity'])
    if status_price <= 0:
        return 0.0, 0
    if status_qty <= 0:
        status_qty = requested_lots
    elif status_qty > requested_lots and status_qty % 1000 == 0:
        status_qty = status_qty / 1000
    return status_price, int(status_qty)


def _safe_diag_attr(obj, name: str) -> str:
    value = getattr(obj, name, None)
    if value is None or value == '':
        return '-'
    text = str(value)
    return text.replace('\n', ' ')[:80]


def _contract_order_diagnostics(contract, order) -> str:
    contract_fields = [
        'code',
        'exchange',
        'category',
        'day_trade',
        'short_selling_balance',
        'margin_trading_balance',
    ]
    order_fields = [
        'action', 'price_type', 'order_type', 'order_lot', 'order_cond',
        'daytrade_short', 'quantity',
    ]
    contract_diag = ','.join(
        f"{field}={_safe_diag_attr(contract, field)}" for field in contract_fields
    )
    order_diag = ','.join(
        f"{field}={_safe_diag_attr(order, field)}" for field in order_fields
    )
    return f"contract[{contract_diag}], order[{order_diag}]"


def shioaji_place_order(
    symbol: str,
    action: str,
    qty_shares: int,
    *,
    intent: str,
) -> Optional[Dict]:
    """
    Place market order via Shioaji (simulation=True).
    qty_shares will be converted to lots (1 lot = 1000 shares).

    `intent` is one of long_entry, long_exit, short_entry, short_exit.
    The short-entry order must be explicitly marked as a day-trade short;
    otherwise Shioaji treats Sell as a cash-lot sale and rejects it when the
    account has no inventory.

    A return with ``order_state == 'UNKNOWN'`` means the broker response was
    not resolved.  It must never be converted into a local position or
    automatically retried by the caller.
    """
    api = get_shioaji_api()
    if api is None:
        logger.error(f"Shioaji 未連線，無法下單 {symbol}")
        return None

    lots = qty_shares // 1000
    if lots <= 0:
        logger.warning(f"{symbol} 股數 {qty_shares} 不足 1 張，跳過下單")
        return None

    if intent not in {'long_entry', 'long_exit', 'short_entry', 'short_exit'}:
        logger.error(f"{symbol} 未知下單意圖 {intent}，拒絕下單")
        return None

    trade = None
    order_id = ''
    try:
        import shioaji as sj
        contract = api.Contracts.Stocks.get(symbol) or api.Contracts.Stocks[symbol]
        if contract is None:
            logger.error(f"{symbol} 找不到合約")
            return None

        is_short = intent in {'short_entry', 'short_exit'}
        order_cond = (
            sj.constant.StockOrderCond.ShortSelling
            if is_short else sj.constant.StockOrderCond.Cash
        )
        order = api.Order(
            price=0,
            quantity=lots,
            action=sj.constant.Action.Buy if action == 'buy' else sj.constant.Action.Sell,
            price_type=sj.constant.StockPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
            order_lot=sj.constant.StockOrderLot.Common,
            order_cond=order_cond,
            daytrade_short=(intent == 'short_entry'),
            account=api.stock_account,
        )

        trade = api.place_order(contract, order)
        order_id = getattr(trade.status, 'id', '') if trade.status else ''

        # Wait for status callback
        time.sleep(0.5)
        api.update_status(api.stock_account)
        updated_status = str(getattr(trade.status, 'status', ''))
        updated_qty = getattr(trade.status, 'order_quantity', lots) if trade.status else lots

        # Keep non-sensitive order context so post-session review can
        # distinguish strategy rejection from broker execution failure.
        if updated_qty == 0 and ('Failed' in updated_status or 'Cancel' in updated_status):
            diagnostics = _contract_order_diagnostics(contract, order)
            logger.warning(
                f"{symbol} 下單被拒絕 "
                f"(action={action}, requested_lots={lots}, order_id={order_id or '-'}, "
                f"status={updated_status}, order_quantity=0, {diagnostics})"
            )
            return None

        logger.info(
            f"{'📈' if action=='buy' else '📉'} 下單: {symbol} {lots}張 | "
            f"intent={intent}, order_cond={getattr(order, 'order_cond', '-')}, "
            f"daytrade_short={getattr(order, 'daytrade_short', False)} | {order_id}"
        )

        # Wait for fill (up to 10s via deal callback). IOC market orders can
        # arrive in multiple deals, so keep polling briefly and use weighted avg.
        filled_price = 0.0
        filled_lots = 0
        for _ in range(20):
            time.sleep(0.5)
            api.update_status(api.stock_account)
            st = str(getattr(trade.status, 'status', ''))
            if 'Filled' in st or 'Deal' in st:
                filled_price, filled_lots = _extract_fill_summary(trade, lots)
                if filled_lots >= lots:
                    break
                continue
            if 'Cancel' in st or 'Failed' in st:
                filled_price, filled_lots = _extract_fill_summary(trade, lots)
                if filled_lots > 0:
                    break
                logger.warning(f"{symbol} 訂單被取消/失敗: {st}")
                return None

        if filled_lots <= 0 or filled_price <= 0:
            # No confirmed deal: do not fabricate a fill from the live quote
            # or requested quantity.  If the terminal status was not known,
            # callers must treat this as an uncertain order and stop retrying.
            final_status = str(getattr(trade.status, 'status', ''))
            if 'Cancel' in final_status or 'Failed' in final_status:
                logger.warning(
                    f"{symbol} 訂單無成交: status={final_status}, "
                    f"intent={intent}, order_id={order_id or '-'}"
                )
                return None
            logger.error(
                f"{symbol} 委託成交狀態未知，禁止當作成交或重送 "
                f"(intent={intent}, order_id={order_id or '-'}, status={final_status})"
            )
            return {
                'order_id': order_id,
                'status': final_status,
                'order_state': 'UNKNOWN',
                'filled_price': 0.0,
                'lots': 0,
                'qty_shares': 0,
            }

        if filled_lots < lots:
            logger.warning(f"{symbol} 部分成交: {filled_lots}/{lots}張 @ {filled_price:.2f}")

        return {
            'order_id': order_id,
            'status': str(getattr(trade.status, 'status', '')),
            'order_state': 'FILLED' if filled_lots >= lots else 'PARTIAL',
            'filled_price': filled_price,
            'lots': filled_lots,
            'qty_shares': filled_lots * 1000,
        }
    except Exception as e:
        logger.error(f"{symbol} 下單失敗: {type(e).__name__}（券商錯誤 payload 已隱去）")
        if trade is not None or order_id:
            return {
                'order_id': order_id,
                'status': 'EXCEPTION_AFTER_SUBMIT',
                'order_state': 'UNKNOWN',
                'filled_price': 0.0,
                'lots': 0,
                'qty_shares': 0,
            }
        return None


def shioaji_short_entry_allowed(
    symbol: str,
    requested_shares: Optional[int] = None,
) -> Tuple[bool, str]:
    """Return whether a new intraday short entry is allowed for this TW stock."""
    api = get_shioaji_api()
    if api is None:
        return False, 'Shioaji 未連線，拒絕空單進場'

    try:
        contract = api.Contracts.Stocks.get(symbol) or api.Contracts.Stocks[symbol]
    except Exception:
        return False, '查詢放空合約失敗，拒絕空單進場'

    if contract is None:
        return False, '找不到合約，拒絕空單進場'

    day_trade = str(getattr(contract, 'day_trade', '') or '')
    short_selling_balance = getattr(contract, 'short_selling_balance', None)

    if day_trade.lower() in {'onlybuy', 'no', 'false'}:
        return False, f'合約 day_trade={day_trade} 不允許當沖放空，拒絕空單進場'

    if short_selling_balance is None:
        return False, '券源餘額未知，拒絕空單進場'

    try:
        available_shares = int(short_selling_balance)
    except (TypeError, ValueError):
        return False, '券源餘額不可解析，拒絕空單進場'

    if available_shares <= 0:
        return False, '券源餘額為 0，拒絕空單進場'
    if requested_shares is not None and available_shares < requested_shares:
        return False, (
            f'券源不足（可用 {available_shares} 股，需求 {requested_shares} 股），'
            '拒絕空單進場'
        )

    return True, ''


# ============================================================================
# K-bar data
# ============================================================================
def get_kbars(symbol: str, interval: str = '5m', period: str = '5d') -> 'object':
    _ensure_deps()
    ticker = f"{symbol}.TW"
    try:
        df = _yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return _pd.DataFrame()
        if isinstance(df.columns, _pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        logger.debug(f"{ticker} yfinance 失敗: {e}")
        return _pd.DataFrame()


# ============================================================================
# Kill Zone
# ============================================================================
def get_kill_zone() -> Optional[str]:
    now_t = _tw_now().time()
    if CONFIG['KZ1_START'] <= now_t <= CONFIG['KZ1_END']:
        return 'kz1_open'
    if CONFIG['KZ2_START'] <= now_t <= CONFIG['KZ2_END']:
        return 'kz2_close'
    return None


def is_market_open() -> bool:
    now_t = _tw_now().time()
    return CONFIG['MARKET_OPEN'] <= now_t <= CONFIG['MARKET_CLOSE']


def is_force_close_time() -> bool:
    return _tw_now().time() >= CONFIG['FORCE_CLOSE']


def is_no_new_trade_time() -> bool:
    return _tw_now().time() >= CONFIG['NO_NEW_TRADE_AFTER']


# ============================================================================
# Heartbeat
# ============================================================================
_last_heartbeat: float = 0.0


def write_heartbeat(positions_count: int = 0) -> None:
    global _last_heartbeat
    now = time.time()
    if now - _last_heartbeat < CONFIG['HEARTBEAT_INTERVAL_SEC']:
        return
    try:
        HEARTBEAT_FILE.write_text(
            json.dumps({
                'ts': datetime.now(TW_TZ).isoformat(),
                'pid': os.getpid(),
                'positions': positions_count,
                'market': 'TW',
            })
        )
        _last_heartbeat = now
    except Exception:
        pass


# ============================================================================
# Main Bot Class
# ============================================================================
class TWDaytradeBot:

    def __init__(self):
        self.notifier = TelegramNotifier(CONFIG['NOTIFY_CHAT_ID'])
        self.risk = RiskManager(
            capital=CONFIG['INITIAL_BALANCE'],
            max_daily_loss_pct=CONFIG['MAX_DAILY_LOSS_PCT'],
            max_positions=CONFIG['MAX_POSITIONS'],
            kelly_fraction=CONFIG['KELLY_FRACTION'],
            max_single_risk_pct=CONFIG['MAX_SINGLE_RISK_PCT'],
            consecutive_loss_limit=CONFIG['CONSECUTIVE_LOSS_LIMIT'],
            re_entry_cooldown_sec=CONFIG['RE_ENTRY_COOLDOWN_SEC'],
        )
        self.recorder = TradeRecorder('tw', DATA_DIR)
        self.smc = SMCEngine()
        # Crash/restart recovery: positions.json is the source of truth for
        # same-day live positions after the process restarts. Cross-day stale
        # positions are ignored by TradeRecorder.load_positions(), otherwise a
        # previous session's orphaned trade can be misread as today's exposure.
        self.positions: Dict[str, Dict] = self.recorder.load_positions()
        if self.positions:
            logger.warning(f"載入未平倉持倉 {len(self.positions)} 筆: {', '.join(self.positions.keys())}")
        self.watchlist: List[str] = list(DEFAULT_WATCHLIST)
        self._shutdown = False
        self._entry_times: Dict[str, float] = {}
        self._daily_winning_trades: Dict[str, float] = {}  # 2026-03-19: 同檔冷卻期追蹤
        self._day_stop_notified: bool = False
        self._reject_log_state: Dict[str, Dict[str, float | int]] = {}
        self._low_score_log_state: Dict[str, Dict[str, float | int]] = {}
        self._order_fail_cooldown_until: Dict[str, float] = {}
        self._order_fail_counts: Dict[str, int] = {}
        # Broker timeout/unknown-status orders are quarantined across restarts.
        # Retrying before reconciliation can duplicate a real fill.
        self._order_status_unknown: set[str] = self._load_unknown_order_quarantine()
        self._opposing_ob_cooldown_until: Dict[Tuple[str, str], float] = {}
        self._kz1_side_stop_cooldown_until: Dict[str, float] = {}

        self._restore_risk_state()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _load_unknown_order_quarantine(self) -> set[str]:
        path = unknown_order_quarantine_path()
        if not path.exists():
            return set()
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            symbols = payload.get('symbols', [])
            if payload.get('date') != _today_str() or not isinstance(symbols, list):
                raise ValueError('invalid quarantine payload')
            quarantined = {str(symbol) for symbol in symbols if str(symbol)}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.error(f"未知委託隔離檔無法讀取，為安全起見不恢復: {exc}")
            return set()
        if quarantined:
            logger.error(
                f"恢復未知委託隔離 {len(quarantined)} 檔: {', '.join(sorted(quarantined))}；"
                "需券商 reconciliation 後才可解除"
            )
        return quarantined

    def _quarantine_unknown_order(self, symbol: str) -> None:
        """Persist the fail-closed marker so a watchdog restart cannot retry it."""
        self._order_status_unknown.add(symbol)
        path = unknown_order_quarantine_path()
        payload = {
            'date': _today_str(),
            'symbols': sorted(self._order_status_unknown),
            'updated_at': _tw_now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        tmp_path = path.with_suffix('.tmp')
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            tmp_path.replace(path)
        except OSError as exc:
            logger.error(f"{symbol}: 未知委託隔離檔寫入失敗，當前程序仍維持隔離: {exc}")

    @staticmethod
    def _parse_trade_time(value: Optional[str]) -> float:
        """交易檔時間字串（台灣時間）→ epoch 秒；失敗回 0。"""
        if not value:
            return 0.0
        try:
            return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=TW_TZ).timestamp()
        except Exception:
            return 0.0

    def _restore_risk_state(self) -> None:
        """
        v2.1.15: 重啟後從當日交易檔重建風控狀態。

        沒有這段的話，盤中重啟（含 watchdog 拉起）會把 daily_pnl、連虧計數、
        同檔停損 ban、獲利後冷卻、KZ1 同方向冷卻全部歸零 —— 等於重啟一次就
        洗掉一次當日風控，watchdog 反而變成風險放大器。
        """
        closed = [
            t for t in self.recorder.get_today_trades()
            if t.get('status') == 'closed'
        ]
        if not closed:
            return

        closed.sort(key=lambda t: t.get('exit_time') or '')
        balance_before = self.risk.current_balance

        for t in closed:
            pnl = float(t.get('pnl') or 0)
            reason = str(t.get('exit_reason') or '')
            reason_lc = reason.lower()
            symbol = t.get('symbol')
            exit_ts = self._parse_trade_time(t.get('exit_time'))

            is_trailing_stop = 'tsl' in reason_lc or '移動停損' in reason
            is_stop_loss = (
                ('stop' in reason_lc or 'sl' in reason_lc or '停損' in reason)
                and pnl < 0
                and not is_trailing_stop
            )

            self.risk.record_trade(pnl, symbol=symbol, is_stop_loss=is_stop_loss)

            if pnl > 0 and not is_stop_loss and symbol and exit_ts:
                self._daily_winning_trades[symbol] = exit_ts

            # KZ1 同方向冷卻：交易檔沒有 kill_zone 欄位，改用出場時間落在
            # KZ1 區間（09:00-10:30）當判準，寧可多冷卻也不要少冷卻。
            side = t.get('side')
            if is_stop_loss and side in {'long', 'short'} and exit_ts:
                exit_dt = datetime.fromtimestamp(exit_ts, TW_TZ)
                if dt_time(9, 0) <= exit_dt.time() <= dt_time(10, 30):
                    until = exit_ts + CONFIG['KZ1_SIDE_STOP_COOLDOWN_SEC']
                    if until > self._kz1_side_stop_cooldown_until.get(side, 0):
                        self._kz1_side_stop_cooldown_until[side] = until

        # current_balance 由 broker 端反映實現損益，避免重複扣一次
        self.risk.current_balance = balance_before

        logger.warning(
            f"重啟恢復風控狀態: 已完成 {len(closed)} 筆 | "
            f"日 PnL={self.risk.daily_pnl:+,.0f} | 連虧={self.risk.consecutive_losses} | "
            f"day_stopped={self.risk.day_stopped}"
        )

    def _handle_signal(self, signum, frame):
        logger.info(f"收到信號 {signum}，開始優雅關閉...")
        self._shutdown = True

    def _graceful_shutdown(self):
        logger.info("優雅關閉：強制平倉所有持倉")
        self.force_close_all("Bot 關閉")
        unsubscribe_all()
        api = get_shioaji_api()
        if api:
            try:
                api.logout()
            except Exception:
                pass
        if self.positions:
            symbols = ', '.join(sorted(self.positions))
            logger.critical(f"Bot 關閉時仍有未平倉持倉: {symbols}；已保留至 positions 檔案待 reconciliation")
            self.notifier.on_error(
                f"收盤/關閉後仍有未平倉持倉 ({symbols})，請查詢券商成交狀態並人工處理", 'TW'
            )
        else:
            self.notifier.on_shutdown('TW', self.risk.trade_count, self.risk.daily_pnl)
            logger.info("Bot 已安全關閉")

    def _log_reject(self, symbol: str, reason: str, *, throttle_sec: int = 300) -> None:
        """Throttle repeated reject warnings to reduce log spam."""
        now = time.time()
        key = f"{symbol}|{reason}"
        state = self._reject_log_state.get(key)
        if state is None or now - float(state['last_ts']) >= throttle_sec:
            repeat = int(state['count']) if state else 0
            suffix = f"（近{throttle_sec//60}分鐘重複 x{repeat + 1}）" if repeat > 0 else ""
            logger.warning(f"{symbol}: {reason}{suffix}")
            self._reject_log_state[key] = {'last_ts': now, 'count': 0}
            return
        state['count'] = int(state['count']) + 1
        state['last_ts'] = now

    def _log_low_score_skip(self, symbol: str, score: int, min_score: int, *, throttle_sec: int = 300) -> None:
        """Throttle repeated low-score debug lines to keep post-mortem readable."""
        now = time.time()
        key = f"{symbol}|{score}|{min_score}"
        state = self._low_score_log_state.get(key)
        if state is None or now - float(state['last_ts']) >= throttle_sec:
            repeat = int(state['count']) if state else 0
            suffix = f"（近{throttle_sec//60}分鐘分數未變 x{repeat + 1}）" if repeat > 0 else ""
            logger.debug(f"{symbol}: 評分 {score} < {min_score}，跳過{suffix}")
            self._low_score_log_state[key] = {'last_ts': now, 'count': 0}
            return
        state['count'] = int(state['count']) + 1
        state['last_ts'] = now

    # -------------------------------------------------------------------------
    # Screener
    # -------------------------------------------------------------------------
    def run_screener(self) -> List[str]:
        """Pre-market screening: return candidates passing basic filters."""
        _ensure_deps()
        candidates = []
        for sym in DEFAULT_WATCHLIST:
            try:
                df = get_kbars(sym, interval='1h', period='5d')
                if df.empty or len(df) < 10:
                    continue
                avg_vol = df['Volume'].tail(20).mean()
                if avg_vol < 500000:
                    continue
                price = float(df['Close'].iloc[-1])
                if price > CONFIG['MAX_STOCK_PRICE']:
                    continue
                candidates.append(sym)
            except Exception:
                pass
        logger.info(f"篩選結果: {len(candidates)} 支通過 / {len(DEFAULT_WATCHLIST)} 支")
        return candidates if candidates else list(DEFAULT_WATCHLIST)

    # -------------------------------------------------------------------------
    # Entry analysis
    # -------------------------------------------------------------------------
    def _analyze_entry(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol for entry. Returns entry dict or None."""
        _ensure_deps()
        try:
            df_5m = get_kbars(symbol, interval='5m', period='2d')
            df_15m = get_kbars(symbol, interval='15m', period='5d')
            df_1h = get_kbars(symbol, interval='1h', period='1mo')

            if df_5m.empty or len(df_5m) < 20:
                return None

            current_price = float(df_5m['Close'].iloc[-1])
            if current_price <= 0:
                return None

            atr = calculate_atr(df_5m)
            vwap_data = VWAPCalculator.calculate(df_5m)
            kill_zone = get_kill_zone()

            result = self.smc.analyze(
                df_5m=df_5m,
                df_15m=df_15m if not df_15m.empty else None,
                df_1h=df_1h if not df_1h.empty else None,
                vwap_data=vwap_data,
                kill_zone=kill_zone,
                atr=atr,
            )

            if result.rejected:
                self._log_reject(symbol, result.reject_reason)
                return None

            # 根據 bias 決定方向 — 雙向策略（v2.1.0）
            if result.bias == 'bullish':
                direction = 'long'
            elif result.bias == 'bearish':
                direction = 'short'
            else:
                # neutral → 不進場
                return None

            # 2026-04-02: 方向一致性硬性檢查，避免 signal 文案與實際方向打架
            has_bullish_signal = any('bullish' in str(s).lower() for s in result.signals)
            has_bearish_signal = any('bearish' in str(s).lower() for s in result.signals)
            if direction == 'long' and has_bearish_signal and not has_bullish_signal:
                self._log_reject(symbol, 'bearish signals only but direction=long，拒絕進場')
                return None
            if direction == 'short' and has_bullish_signal and not has_bearish_signal:
                self._log_reject(symbol, 'bullish signals only but direction=short，拒絕進場')
                return None

            if direction == 'short':
                short_allowed, short_reject_reason = shioaji_short_entry_allowed(symbol)
                if not short_allowed:
                    self._log_reject(symbol, short_reject_reason)
                    return None

            if kill_zone == 'kz1_open':
                side_cooldown_until = self._kz1_side_stop_cooldown_until.get(direction, 0)
                if time.time() < side_cooldown_until:
                    self._log_reject(
                        symbol,
                        f'KZ1 {direction} initial SL cooldown active，拒絕進場',
                        throttle_sec=120,
                    )
                    return None

            # 2026-05-07: 15M order block is an execution zone, not just decorative text.
            # If it points against the planned side, the setup is trading into an opposing
            # supply/demand zone; do not let 1H/5M alignment hide that conflict.
            has_bullish_15m_ob = any('15m ob(bullish)' in str(s).lower() for s in result.signals)
            has_bearish_15m_ob = any('15m ob(bearish)' in str(s).lower() for s in result.signals)
            opposing_ob_key = (symbol, direction)
            opposing_ob_cooldown_until = self._opposing_ob_cooldown_until.get(opposing_ob_key, 0)
            if time.time() < opposing_ob_cooldown_until:
                self._log_reject(
                    symbol,
                    f'15M opposing OB cooldown active for {direction} setup，拒絕進場',
                    throttle_sec=120,
                )
                return None
            if direction == 'long' and has_bearish_15m_ob:
                self._opposing_ob_cooldown_until[(symbol, direction)] = (
                    time.time() + CONFIG['OPPOSING_OB_COOLDOWN_SEC']
                )
                self._log_reject(symbol, '15M bearish OB opposes long setup，拒絕進場')
                return None
            if direction == 'short' and has_bullish_15m_ob:
                self._opposing_ob_cooldown_until[(symbol, direction)] = (
                    time.time() + CONFIG['OPPOSING_OB_COOLDOWN_SEC']
                )
                self._log_reject(symbol, '15M bullish OB opposes short setup，拒絕進場')
                return None

            # Score threshold
            now_t = _tw_now().time()
            if now_t < CONFIG['OPENING_GRACE_END']:
                min_score = 95
            elif kill_zone == 'kz1_open':
                min_score = CONFIG['KZ1_MIN_SCORE']
            elif kill_zone == 'kz2_close':
                min_score = CONFIG['KZ2_MIN_SCORE']  # 2026-03-19: KZ2時段門檻提高至98
            else:
                min_score = CONFIG['MIN_SCORE']

            if result.score < min_score:
                self._log_low_score_skip(symbol, result.score, min_score)
                return None

            # 2026-03-20: KZ1 & KZ2時段均拒絕weak VWAP rejection（0% 勝率）
            # 2026-05-28: weak VWAP bounce 即使 0 分，仍可能被其他 SMC signals 堆到 Score 100；
            # 今日 2891 KZ1 weak bounce 再次停損，故 KZ 內所有 weak VWAP interaction 一律硬拒絕。
            has_weak_vwap_interaction = any(
                'vwap' in str(s).lower() and 'weak' in str(s).lower()
                for s in result.signals
            )
            if kill_zone in ['kz1_open', 'kz2_close'] and has_weak_vwap_interaction:
                self._log_reject(symbol, f'{kill_zone}時段含weak VWAP interaction，拒絕進場')
                return None

            # 2026-04-17 / 2026-04-24: KZ2 尾盤不只要 strong VWAP interaction，
            # 還要保留足夠時間讓 setup 發酵。12:35 後的新倉 edge 太容易被 13:00 硬截止吃掉。
            # 2026-05-29: KZ1 在 09:05 開盤寬限後，只有「VWAP 上/下方」或沒有明確 strong
            # interaction 的 setup 歷史績效明顯偏差；要求 strong bounce/rejection 才放行。
            # 2026-09-22: 2609 在 09:00 寬限內僅靠 VWAP 上方與 SMC 堆到 Score 100 後
            # 3 分鐘 hit initial SL；取消 KZ1 strong VWAP 的開盤寬限。
            has_vwap_signal = any('vwap' in str(s).lower() for s in result.signals)
            has_strong_vwap_interaction = any(
                'vwap' in str(s).lower() and 'strong' in str(s).lower()
                for s in result.signals
            )
            if kill_zone == 'kz1_open' and not has_strong_vwap_interaction:
                self._log_reject(symbol, 'KZ1缺乏strong VWAP interaction，拒絕進場')
                return None
            if kill_zone == 'kz2_close' and now_t >= dt_time(12, 30):
                if now_t >= CONFIG['NO_NEW_TRADE_AFTER']:
                    self._log_reject(symbol, '接近13:00硬截止，拒絕尾盤新倉')
                    return None
                if has_vwap_signal and not has_strong_vwap_interaction:
                    self._log_reject(symbol, 'KZ2最後30分鐘缺乏strong VWAP interaction，拒絕進場')
                    return None

            # 2026-03-24: 檢查 1H 趨勢方向 — 防止逆勢交易
            # 提取 1H 信號，判斷 1H BOS/CHoCH 方向
            h1_signal = next((s for s in result.signals if '1H' in s), None)
            if h1_signal:
                # 例如："1H BOS bearish", "1H CHOCH bullish"
                h1_is_bearish = 'bearish' in h1_signal.lower()
                h1_is_bullish = 'bullish' in h1_signal.lower()

                if direction == 'long' and h1_is_bearish:
                    self._log_reject(symbol, f'1H bearish 但尋找多單（Score={result.score}, 逆勢），拒絕進場')
                    return None
                if direction == 'short' and h1_is_bullish:
                    self._log_reject(symbol, f'1H bullish 但尋找空單（Score={result.score}, 逆勢），拒絕進場')
                    return None

            # 2026-03-24: 調整 KZ 時段停損寬度 — 最少保留 0.4% 或 ATR*2.5，取較大值
            # 原因：KZ 時段波動率大，過緊的停損容易被掃盤（如 2615 只有 0.13 點）
            # 2026-04-23: 最小停損寬度必須依標的價位分級；固定 0.4% 對低價金融股仍太緊，
            # 對高價股卻又可能太寬。改成 tiered floor，避免 2882/2886 這種午盤單被 1~2 tick 掃掉。
            # v2.1.0: 停損/獲利方向依 direction 反向計算
            base_sl_width = atr * CONFIG['ATR_STOP_MULT']
            if current_price < 50:
                min_sl_pct = 0.006  # 低價股至少 0.6%
            elif current_price < 80:
                min_sl_pct = 0.005  # 中價股至少 0.5%
            else:
                min_sl_pct = 0.004  # 高價股維持 0.4%
            min_sl_width = current_price * min_sl_pct
            if kill_zone in ['kz1_open', 'kz2_close']:
                sl_width = max(base_sl_width, min_sl_width)
                logger.debug(
                    f"{symbol} (KZ時段): SL widened to {sl_width:.2f} "
                    f"(base={base_sl_width:.2f}, min_pct={min_sl_width:.2f}, floor={min_sl_pct:.3%})"
                )
            else:
                sl_width = base_sl_width

            if direction == 'long':
                stop_price = current_price - sl_width
                take_profit = current_price + atr * CONFIG['ATR_TP_MULT']
            else:  # short
                stop_price = current_price + sl_width
                take_profit = current_price - atr * CONFIG['ATR_TP_MULT']

            qty = self.risk.calculate_position_size_tw(
                entry_price=current_price,
                stop_price=stop_price,
                smc_score=result.score,
            )
            if qty <= 0:
                logger.warning(f"{symbol}: 部位計算為 0，跳過")
                return None

            return {
                'symbol': symbol,
                'price': current_price,
                'qty': qty,
                'score': result.score,
                'bias': result.bias,
                'direction': direction,
                'signals': result.signals,
                'stop_price': stop_price,
                'take_profit': take_profit,
                'atr': atr,
                'kill_zone': kill_zone,
            }
        except Exception as e:
            logger.error(f"{symbol} 分析失敗: {e}")
            return None

    def _enter_position(self, entry: Dict) -> bool:
        """Execute entry order. Returns True if filled."""
        symbol = entry['symbol']

        ok, reason = self.risk.can_open_position(len(self.positions))
        if not ok:
            logger.info(f"{symbol}: 跳過進場 — {reason}")
            return False

        now_ts = time.time()

        if symbol in self._order_status_unknown:
            self._log_reject(
                symbol,
                '先前委託成交狀態未知，等待券商 reconciliation，禁止重送',
                throttle_sec=1800,
            )
            return False

        order_fail_count = self._order_fail_counts.get(symbol, 0)
        if order_fail_count >= CONFIG['ORDER_FAIL_DAILY_LIMIT']:
            self._log_reject(
                symbol,
                f"券商拒單已達 {order_fail_count} 次，今日停止重送 entry order",
                throttle_sec=1800,
            )
            return False

        order_fail_cooldown_until = self._order_fail_cooldown_until.get(symbol, 0)
        if now_ts < order_fail_cooldown_until:
            logger.debug(f"{symbol}: 下單失敗冷卻中")
            return False

        last_entry = self._entry_times.get(symbol, 0)
        if now_ts - last_entry < CONFIG['ENTRY_COOLDOWN_SEC']:
            logger.debug(f"{symbol}: 進場冷卻中")
            return False

        # 2026-03-19: 同檔已獲利出場後2小時內拒絕重新進場
        last_winning_exit = self._daily_winning_trades.get(symbol, 0)
        if time.time() - last_winning_exit < 2 * 3600:  # 2小時
            self._log_reject(symbol, '已獲利出場，2小時冷卻期中，拒絕進場', throttle_sec=1800)
            return False

        # v2.1.20: 依方向明確指定 entry intent，避免 short Sell 被當成 Cash。
        entry_direction = entry.get('direction', 'long')
        order_action = 'buy' if entry_direction == 'long' else 'sell'
        order_intent = 'long_entry' if entry_direction == 'long' else 'short_entry'
        if entry_direction == 'short':
            short_allowed, short_reject_reason = shioaji_short_entry_allowed(
                symbol, entry['qty']
            )
            if not short_allowed:
                self._log_reject(symbol, short_reject_reason)
                return False
        result = shioaji_place_order(
            symbol, order_action, entry['qty'], intent=order_intent
        )
        if result and result.get('order_state') == 'UNKNOWN':
            self._quarantine_unknown_order(symbol)
            logger.error(f"{symbol}: 委託狀態未知，已隔離該檔並禁止重送")
            self.notifier.on_error(
                f"{symbol} 委託狀態未知，已停止重送，請查詢券商成交狀態", 'TW'
            )
            return False
        if result is None:
            logger.warning(
                f"{symbol}: 下單失敗 "
                f"(direction={entry.get('direction', 'long')}, score={entry.get('score')}, "
                f"price={entry.get('price', 0):.2f}, stop={entry.get('stop_price', 0):.2f}, "
                f"tp={entry.get('take_profit', 0):.2f}, qty={entry.get('qty', 0)})"
            )
            fail_count = self._order_fail_counts.get(symbol, 0) + 1
            self._order_fail_counts[symbol] = fail_count
            cooldown_until = now_ts + max(CONFIG['ENTRY_COOLDOWN_SEC'], 15 * 60)
            self._entry_times[symbol] = now_ts  # avoid immediate re-fire spam after rejected order
            self._order_fail_cooldown_until[symbol] = cooldown_until
            if fail_count >= CONFIG['ORDER_FAIL_DAILY_LIMIT']:
                logger.warning(
                    f"{symbol}: 券商拒單累計 {fail_count} 次，今日停止重送 entry order"
                )
            return False

        self._order_fail_cooldown_until.pop(symbol, None)
        self._order_fail_counts.pop(symbol, None)

        if result.get('order_state') not in {'FILLED', 'PARTIAL'}:
            logger.error(f"{symbol}: 無有效成交狀態，保留空手並禁止建立持倉")
            return False
        filled_price = result['filled_price']
        qty_shares = result['qty_shares']
        if filled_price <= 0 or qty_shares <= 0:
            logger.error(f"{symbol}: 成交資料不完整，禁止建立持倉")
            return False

        # 2026-05-04: Market fills can slip away from the analysis snapshot price.
        # Keep the intended stop / TP distances, but anchor them to the actual fill;
        # otherwise a short filled above the snapshot can inherit a stop below entry
        # and immediately self-trigger as a bogus break-even SL.
        planned_price = entry['price']
        stop_distance = abs(planned_price - entry['stop_price'])
        tp_distance = abs(entry['take_profit'] - planned_price)
        if entry_direction == 'long':
            actual_stop_price = filled_price - stop_distance
            actual_take_profit = filled_price + tp_distance
        else:
            actual_stop_price = filled_price + stop_distance
            actual_take_profit = filled_price - tp_distance

        trade_id = self.recorder.on_entry(
            symbol=symbol,
            entry_price=filled_price,
            qty=qty_shares,
            score=entry['score'],
            signals=entry['signals'],
            stop_price=actual_stop_price,
            take_profit=actual_take_profit,
            side=entry_direction,
        )

        self.positions[symbol] = {
            'trade_id': trade_id,
            'entry_price': filled_price,
            'qty': qty_shares,
            'initial_stop_price': actual_stop_price,
            'stop_price': actual_stop_price,
            'take_profit': actual_take_profit,
            'atr': entry['atr'],
            'direction': entry_direction,
            'trailing_stop_active': False,
            'highest_price': filled_price,
            'lowest_price': filled_price,
            'entry_time': time.time(),
            'kill_zone': entry.get('kill_zone'),
        }
        self._entry_times[symbol] = time.time()
        self.recorder.update_positions(self.positions)

        self.notifier.on_entry(
            symbol=symbol,
            price=filled_price,
            qty=qty_shares,
            score=entry['score'],
            signals=entry['signals'],
            stop=actual_stop_price,
            tp=actual_take_profit,
            market='TW',
        )
        direction_label = '做多' if entry_direction == 'long' else '做空'
        logger.info(
            f"✅ {direction_label}進場: {symbol} @ {filled_price:.2f} × {qty_shares}股 "
            f"| Score={entry['score']} | 停損={actual_stop_price:.2f}"
        )
        return True

    def _exit_position(self, symbol: str, reason: str) -> None:
        """Execute exit order for a position."""
        pos = self.positions.get(symbol)
        if not pos:
            return
        if symbol in self._order_status_unknown:
            logger.error(f"{symbol}: 先前出場委託狀態未知，禁止重送出場單")
            return

        current_price = get_live_price(symbol)
        if current_price <= 0:
            current_price = pos['entry_price']

        # v2.1.0: 依持倉方向決定平倉動作
        pos_direction = pos.get('direction', 'long')
        close_action = 'sell' if pos_direction == 'long' else 'buy'
        position_qty = pos['qty']
        close_intent = 'long_exit' if pos_direction == 'long' else 'short_exit'
        result = shioaji_place_order(
            symbol, close_action, position_qty, intent=close_intent
        )
        if result and result.get('order_state') == 'UNKNOWN':
            self._quarantine_unknown_order(symbol)
            logger.error(f"{symbol}: 出場委託狀態未知，保留原持倉並停止重送")
            self.notifier.on_error(
                f"{symbol} 出場委託狀態未知，持倉保留，請查詢券商成交狀態", 'TW'
            )
            return
        if result is None:
            logger.warning(f"{symbol}: 出場下單失敗，保留原持倉 | reason={reason}")
            self.notifier.on_error(f"{symbol} 出場下單失敗，持倉仍保留", 'TW')
            return

        if result.get('order_state') not in {'FILLED', 'PARTIAL'}:
            logger.warning(f"{symbol}: 出場沒有有效成交狀態，保留原持倉")
            return
        filled_qty = min(result.get('qty_shares', 0), position_qty)
        if result['filled_price'] > 0:
            current_price = result['filled_price']
        if filled_qty <= 0:
            logger.warning(f"{symbol}: 出場未成交，保留原持倉")
            return

        # v2.1.0: PnL 依方向計算
        if pos_direction == 'long':
            pnl = (current_price - pos['entry_price']) * filled_qty
        else:  # short
            pnl = (pos['entry_price'] - current_price) * filled_qty
        reason_lc = reason.lower()
        is_trailing_stop = 'tsl' in reason_lc or '移動停損' in reason
        is_stop_loss = (
            ('stop' in reason_lc or 'sl' in reason_lc or '停損' in reason)
            and pnl < 0
            and not is_trailing_stop
        )

        # 2026-03-19: 記錄當日獲利的交易，用於冷卻期追蹤
        if pnl > 0 and not is_stop_loss:
            self._daily_winning_trades[symbol] = time.time()
        if (
            is_stop_loss
            and pos.get('kill_zone') == 'kz1_open'
            and pos_direction in {'long', 'short'}
        ):
            self._kz1_side_stop_cooldown_until[pos_direction] = (
                time.time() + CONFIG['KZ1_SIDE_STOP_COOLDOWN_SEC']
            )
            logger.warning(
                f"KZ1 {pos_direction} initial SL → 同方向冷卻 "
                f"{CONFIG['KZ1_SIDE_STOP_COOLDOWN_SEC']} 秒"
            )

        trade_id = pos.get('trade_id', f"{symbol}_unknown")
        is_partial_exit = filled_qty < position_qty
        self.recorder.on_exit(
            trade_id,
            current_price,
            pnl,
            reason,
            exit_qty=filled_qty if is_partial_exit else None,
        )
        self.risk.record_trade(pnl, symbol=symbol, is_stop_loss=is_stop_loss)

        if self.risk.day_stopped and not self._day_stop_notified:
            self.notifier.on_day_stop('TW', self.risk.day_stop_reason)
            self._day_stop_notified = True

        self.notifier.on_exit(
            symbol=symbol,
            entry_price=pos['entry_price'],
            exit_price=current_price,
            qty=filled_qty,
            pnl=pnl,
            reason=reason,
            market='TW',
        )
        direction_label = '多' if pos_direction == 'long' else '空'
        if is_partial_exit:
            pos['qty'] = position_qty - filled_qty
            self.recorder.update_positions(self.positions)
            logger.warning(
                f"{symbol} {direction_label}單部分出場: {filled_qty}/{position_qty}股 "
                f"@ {current_price:.2f} | PnL={pnl:+,.0f} | 剩餘={pos['qty']}股 | {reason}"
            )
            return

        logger.info(
            f"{'✅' if pnl >= 0 else '❌'} {direction_label}單出場: {symbol} @ {current_price:.2f} "
            f"| PnL={pnl:+,.0f} | {reason}"
        )

        del self.positions[symbol]
        self.recorder.update_positions(self.positions)

    def force_close_all(self, reason: str = "強制平倉") -> None:
        if not self.positions:
            return
        logger.warning(f"強制平倉 ({len(self.positions)} 個): {reason}")
        for symbol in list(self.positions.keys()):
            try:
                self._exit_position(symbol, reason)
            except Exception as e:
                logger.error(f"{symbol} 強制平倉失敗: {e}")
                self.notifier.on_error(f"{symbol} 強制平倉失敗: {e}", 'TW')

    # -------------------------------------------------------------------------
    # Monitor positions
    # -------------------------------------------------------------------------
    def _monitor_positions(self) -> None:
        for symbol in list(self.positions.keys()):
            try:
                pos = self.positions[symbol]
                current_price = get_live_price(symbol)
                if current_price <= 0:
                    continue

                # v2.1.0: Trailing stop — 依方向反向追蹤
                pos_direction = pos.get('direction', 'long')
                if pos_direction == 'long':
                    highest = pos.get('highest_price', pos['entry_price'])
                    if current_price > highest:
                        pos['highest_price'] = current_price
                        new_stop = current_price - pos.get('atr', 1.0) * CONFIG['ATR_STOP_MULT']
                        if new_stop > pos.get('stop_price', 0):
                            pos['stop_price'] = new_stop
                            pos['trailing_stop_active'] = True
                            self.recorder.update_positions(self.positions)
                            logger.debug(f"{symbol} 多單移動停損: {new_stop:.2f}")

                    if current_price <= pos.get('stop_price', 0):
                        reason = '多單停損(SL)'
                        if pos.get('trailing_stop_active') or pos.get('stop_price', 0) > pos['entry_price']:
                            reason = '多單移動停損(TSL)'
                        self._exit_position(symbol, reason)
                    elif current_price >= pos.get('take_profit', float('inf')):
                        self._exit_position(symbol, '多單獲利了結(TP)')
                else:  # short
                    lowest = pos.get('lowest_price', pos['entry_price'])
                    if current_price < lowest:
                        pos['lowest_price'] = current_price
                        new_stop = current_price + pos.get('atr', 1.0) * CONFIG['ATR_STOP_MULT']
                        if new_stop < pos.get('stop_price', float('inf')):
                            pos['stop_price'] = new_stop
                            pos['trailing_stop_active'] = True
                            self.recorder.update_positions(self.positions)
                            logger.debug(f"{symbol} 空單移動停損: {new_stop:.2f}")

                    if current_price >= pos.get('stop_price', float('inf')):
                        reason = '空單停損(SL)'
                        if pos.get('trailing_stop_active') or pos.get('stop_price', float('inf')) < pos['entry_price']:
                            reason = '空單移動停損(TSL)'
                        self._exit_position(symbol, reason)
                    elif current_price <= pos.get('take_profit', 0):
                        self._exit_position(symbol, '空單獲利了結(TP)')
            except Exception as e:
                logger.error(f"{symbol} 監控持倉失敗: {e}")
                continue

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------
    def run(self) -> None:
        """Main daemon mode with watchdog loop — NEVER crashes permanently."""
        restart_count = 0
        while not self._shutdown:
            try:
                restart_count += 1
                if restart_count > 1:
                    wait = min(60, restart_count * 5)
                    logger.warning(f"Bot 重啟中（第{restart_count}次），等待 {wait}s...")
                    self.notifier.on_restart('TW', '異常重啟', restart_count)
                    time.sleep(wait)

                self._run_loop()

                if self._shutdown:
                    break

                logger.info("交易日結束，Bot 停止")
                break

            except KeyboardInterrupt:
                logger.info("收到 Ctrl+C")
                self._shutdown = True
            except Exception as e:
                logger.error(f"主迴圈崩潰: {e}\n{traceback.format_exc()}")
                self.notifier.on_error(f"主迴圈崩潰: {e}", 'TW')
                if self._shutdown:
                    break

        self._graceful_shutdown()

    def _run_loop(self) -> None:
        """Inner trading loop. Returns when market closes."""
        _ensure_deps()
        logger.info("=" * 60)
        logger.info(f"TW Bot v{__version__} 啟動 — {_today_str()}")
        logger.info("=" * 60)

        screened = self.run_screener()
        if screened:
            self.watchlist = screened

        subscribe_quotes(self.watchlist)
        self.notifier.on_startup('TW', len(self.watchlist), CONFIG['INITIAL_BALANCE'])

        last_scan = 0.0
        last_monitor = 0.0
        last_quote_reconnect_check = 0.0

        while not self._shutdown:
            write_heartbeat(len(self.positions))

            if is_force_close_time():
                if self.positions:
                    logger.warning("⏰ 13:00 強制平倉")
                    self.force_close_all("13:00 硬截止強制平倉")
                logger.info("🏁 收盤，Bot 停止")
                return

            if self.risk.day_stopped:
                if self.positions:
                    self.force_close_all(self.risk.day_stop_reason)
                time.sleep(30)
                continue

            now_ts = time.time()

            # Startup login may time out.  Keep retrying subscription at a
            # bounded cadence so later broker recovery restores tick data
            # instead of silently falling back to delayed candles all day.
            if now_ts - last_quote_reconnect_check >= CONFIG['BROKER_RETRY_INTERVAL_SEC']:
                if len(_subscribed_symbols) < len(self.watchlist):
                    subscribe_quotes(self.watchlist)
                last_quote_reconnect_check = now_ts

            if self.positions and (now_ts - last_monitor >= CONFIG['MONITOR_INTERVAL_SEC']):
                try:
                    self._monitor_positions()
                except Exception as e:
                    logger.error(f"監控持倉失敗: {e}")
                last_monitor = now_ts

            if (
                not is_no_new_trade_time()
                and is_market_open()
                and get_kill_zone() is not None
                and len(self.positions) < CONFIG['MAX_POSITIONS']
                and (now_ts - last_scan >= CONFIG['SCAN_INTERVAL_SEC'])
            ):
                for symbol in self.watchlist:
                    if symbol in self.positions:
                        continue
                    ok, reason = self.risk.can_trade(symbol)
                    if not ok:
                        continue
                    entry = self._analyze_entry(symbol)
                    if entry:
                        self._enter_position(entry)
                        if len(self.positions) >= CONFIG['MAX_POSITIONS']:
                            break
                last_scan = now_ts

            time.sleep(5)


# ============================================================================
# Subcommands
# ============================================================================
def cmd_screener():
    _ensure_deps()
    bot = TWDaytradeBot.__new__(TWDaytradeBot)
    bot.watchlist = list(DEFAULT_WATCHLIST)
    # Minimal init for screener
    bot.notifier = TelegramNotifier(CONFIG['NOTIFY_CHAT_ID'])
    results = bot.run_screener()
    print(f"\n台股盤前篩選結果 ({_today_str()}):")
    for sym in results:
        print(f"  ✓ {sym}")
    print(f"\n共 {len(results)} 支通過篩選")


def cmd_log():
    recorder = TradeRecorder('tw', DATA_DIR)
    recorder.print_summary()


def cmd_status():
    recorder = TradeRecorder('tw', DATA_DIR)
    positions = recorder.load_positions()
    print(f"\n台股持倉狀態 ({_today_str()}):")
    if not positions:
        print("  目前無持倉")
    else:
        for sym, pos in positions.items():
            price = get_live_price(sym)
            direction = pos.get('direction', 'long')
            if direction == 'short':
                pnl = (pos.get('entry_price', 0) - price) * pos.get('qty', 0)
            else:
                pnl = (price - pos.get('entry_price', 0)) * pos.get('qty', 0)
            print(f"  {sym}: 進場={pos.get('entry_price', 0):.2f} 現價={price:.2f} PnL={pnl:+,.0f}")
    recorder.print_summary()


# ============================================================================
# Entry point
# ============================================================================
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'

    if cmd == 'screener':
        cmd_screener()
    elif cmd == 'log':
        cmd_log()
    elif cmd == 'status':
        cmd_status()
    elif cmd == 'run':
        bot = TWDaytradeBot()
        bot.run()
    else:
        print(f"用法: {sys.argv[0]} [run|screener|log|status]")
        sys.exit(1)


if __name__ == '__main__':
    main()
