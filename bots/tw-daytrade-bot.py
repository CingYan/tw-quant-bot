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

⚠️ 模擬交易專用 — simulation=True，禁止真實下單
"""

__version__ = "2.0.1"

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
    'MIN_SCORE': 70,
    'KZ1_MIN_SCORE': 90,
    'FVG_SATURATION_THRESHOLD': 7,
    'CONSECUTIVE_LOSS_LIMIT': 3,
    'HEARTBEAT_INTERVAL_SEC': 30,
    'NOTIFY_CHAT_ID': 'YOUR_TELEGRAM_CHAT_ID',
    'SCAN_INTERVAL_SEC': 30,
    'MONITOR_INTERVAL_SEC': 20,
    # TW market hours
    'MARKET_OPEN': dt_time(9, 0),
    'MARKET_CLOSE': dt_time(13, 30),
    'KZ1_START': dt_time(9, 0),
    'KZ1_END': dt_time(10, 0),
    'KZ2_START': dt_time(12, 30),
    'KZ2_END': dt_time(13, 20),
    'NO_NEW_TRADE_AFTER': dt_time(13, 0),
    'FORCE_CLOSE': dt_time(13, 20),
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
BASE_DIR = Path('/path/to/project')
CRED_FILE = BASE_DIR / 'credentials' / 'credentials.json'
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
_live_prices: Dict[str, float] = {}
_live_prices_ts: Dict[str, float] = {}
_subscribed_symbols: set = set()


def get_shioaji_api():
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


def shioaji_place_order(symbol: str, action: str, qty_shares: int) -> Optional[Dict]:
    """
    Place market order via Shioaji (simulation=True).
    qty_shares will be converted to lots (1 lot = 1000 shares).

    v2 fix: Don't reject on order_quantity=0 from initial status callback.
    Only reject on explicit Failed/Cancelled after update_status.
    """
    api = get_shioaji_api()
    if api is None:
        logger.error(f"Shioaji 未連線，無法下單 {symbol}")
        return None

    lots = qty_shares // 1000
    if lots <= 0:
        logger.warning(f"{symbol} 股數 {qty_shares} 不足 1 張，跳過下單")
        return None

    try:
        import shioaji as sj
        contract = api.Contracts.Stocks.get(symbol) or api.Contracts.Stocks[symbol]
        if contract is None:
            logger.error(f"{symbol} 找不到合約")
            return None

        order = api.Order(
            price=0,
            quantity=lots,
            action=sj.constant.Action.Buy if action == 'buy' else sj.constant.Action.Sell,
            price_type=sj.constant.StockPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
            order_lot=sj.constant.StockOrderLot.Common,
            account=api.stock_account,
        )

        trade = api.place_order(contract, order)
        order_id = getattr(trade.status, 'id', '') if trade.status else ''

        # Wait for status callback
        time.sleep(0.5)
        api.update_status(api.stock_account)
        updated_status = str(getattr(trade.status, 'status', ''))
        updated_qty = getattr(trade.status, 'order_quantity', lots) if trade.status else lots

        # Only reject on explicit failure with qty=0
        if updated_qty == 0 and ('Failed' in updated_status or 'Cancel' in updated_status):
            logger.warning(f"{symbol} 下單被拒絕 (status={updated_status}, qty=0)")
            return None

        logger.info(f"{'📈' if action=='buy' else '📉'} 下單: {symbol} {lots}張 | {order_id}")

        # Wait for fill (up to 10s via deal callback)
        filled_price = 0.0
        for _ in range(20):
            time.sleep(0.5)
            api.update_status(api.stock_account)
            st = str(getattr(trade.status, 'status', ''))
            if 'Filled' in st or 'Deal' in st:
                filled_price = float(getattr(trade.status, 'deal_price', 0) or 0)
                if filled_price <= 0:
                    filled_price = float(getattr(trade.status, 'modified_price', 0) or 0)
                break
            if 'Cancel' in st or 'Failed' in st:
                logger.warning(f"{symbol} 訂單被取消/失敗: {st}")
                return None

        if filled_price <= 0:
            filled_price = get_live_price(symbol)

        return {
            'order_id': order_id,
            'status': str(getattr(trade.status, 'status', '')),
            'filled_price': filled_price,
            'lots': lots,
            'qty_shares': lots * 1000,
        }
    except Exception as e:
        logger.error(f"{symbol} 下單失敗: {e}")
        return None


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
        self.positions: Dict[str, Dict] = {}
        self.watchlist: List[str] = list(DEFAULT_WATCHLIST)
        self._shutdown = False
        self._entry_times: Dict[str, float] = {}
        self._day_stop_notified: bool = False

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

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
        self.notifier.on_shutdown('TW', self.risk.trade_count, self.risk.daily_pnl)
        logger.info("Bot 已安全關閉")

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
                logger.warning(f"{symbol}: {result.reject_reason}")
                return None

            # Score threshold
            now_t = _tw_now().time()
            if now_t < CONFIG['OPENING_GRACE_END']:
                min_score = 95
            elif kill_zone == 'kz1_open':
                min_score = CONFIG['KZ1_MIN_SCORE']
            else:
                min_score = CONFIG['MIN_SCORE']

            if result.score < min_score:
                logger.debug(f"{symbol}: 評分 {result.score} < {min_score}，跳過")
                return None

            if result.bias == 'neutral':
                return None

            stop_price = current_price - atr * CONFIG['ATR_STOP_MULT']
            take_profit = current_price + atr * CONFIG['ATR_TP_MULT']

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

        last_entry = self._entry_times.get(symbol, 0)
        if time.time() - last_entry < CONFIG['ENTRY_COOLDOWN_SEC']:
            logger.debug(f"{symbol}: 進場冷卻中")
            return False

        result = shioaji_place_order(symbol, 'buy', entry['qty'])
        if result is None:
            logger.warning(f"{symbol}: 下單失敗")
            return False

        filled_price = result['filled_price']
        qty_shares = result['qty_shares']
        if filled_price <= 0:
            filled_price = entry['price']

        trade_id = self.recorder.on_entry(
            symbol=symbol,
            entry_price=filled_price,
            qty=qty_shares,
            score=entry['score'],
            signals=entry['signals'],
            stop_price=entry['stop_price'],
            take_profit=entry['take_profit'],
        )

        self.positions[symbol] = {
            'trade_id': trade_id,
            'entry_price': filled_price,
            'qty': qty_shares,
            'stop_price': entry['stop_price'],
            'take_profit': entry['take_profit'],
            'atr': entry['atr'],
            'highest_price': filled_price,
            'entry_time': time.time(),
        }
        self._entry_times[symbol] = time.time()
        self.recorder.update_positions(self.positions)

        self.notifier.on_entry(
            symbol=symbol,
            price=filled_price,
            qty=qty_shares,
            score=entry['score'],
            signals=entry['signals'],
            stop=entry['stop_price'],
            tp=entry['take_profit'],
            market='TW',
        )
        logger.info(
            f"✅ 進場: {symbol} @ {filled_price:.2f} × {qty_shares}股 "
            f"| Score={entry['score']} | 停損={entry['stop_price']:.2f}"
        )
        return True

    def _exit_position(self, symbol: str, reason: str) -> None:
        """Execute exit order for a position."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        current_price = get_live_price(symbol)
        if current_price <= 0:
            current_price = pos['entry_price']

        result = shioaji_place_order(symbol, 'sell', pos['qty'])
        if result and result['filled_price'] > 0:
            current_price = result['filled_price']

        pnl = (current_price - pos['entry_price']) * pos['qty']
        is_stop_loss = 'stop' in reason.lower() or 'sl' in reason.lower()

        trade_id = pos.get('trade_id', f"{symbol}_unknown")
        self.recorder.on_exit(trade_id, current_price, pnl, reason)
        self.risk.record_trade(pnl, symbol=symbol, is_stop_loss=is_stop_loss)

        if self.risk.day_stopped and not self._day_stop_notified:
            self.notifier.on_day_stop('TW', self.risk.day_stop_reason)
            self._day_stop_notified = True

        self.notifier.on_exit(
            symbol=symbol,
            entry_price=pos['entry_price'],
            exit_price=current_price,
            qty=pos['qty'],
            pnl=pnl,
            reason=reason,
            market='TW',
        )
        logger.info(
            f"{'✅' if pnl >= 0 else '❌'} 出場: {symbol} @ {current_price:.2f} "
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

                # Trailing stop
                highest = pos.get('highest_price', pos['entry_price'])
                if current_price > highest:
                    pos['highest_price'] = current_price
                    new_stop = current_price - pos.get('atr', 1.0) * CONFIG['ATR_STOP_MULT']
                    if new_stop > pos.get('stop_price', 0):
                        pos['stop_price'] = new_stop
                        logger.debug(f"{symbol} 移動停損: {new_stop:.2f}")

                if current_price <= pos.get('stop_price', 0):
                    self._exit_position(symbol, '停損(SL)')
                elif current_price >= pos.get('take_profit', float('inf')):
                    self._exit_position(symbol, '獲利了結(TP)')
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

        while not self._shutdown:
            write_heartbeat(len(self.positions))

            if is_force_close_time():
                if self.positions:
                    logger.warning("⏰ 13:20 強制平倉")
                    self.force_close_all("收盤強制平倉")
                self.notifier.on_shutdown('TW', self.risk.trade_count, self.risk.daily_pnl)
                logger.info("🏁 收盤，Bot 停止")
                return

            if self.risk.day_stopped:
                if self.positions:
                    self.force_close_all(self.risk.day_stop_reason)
                time.sleep(30)
                continue

            now_ts = time.time()

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
