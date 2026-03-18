# -*- coding: utf-8 -*-
"""
Trade Recorder v2.0 — Immediate trade logging

Writes trade to disk IMMEDIATELY on entry (partial record).
Updates with exit info when position closes.
Also maintains positions.json for crash recovery.

File: data/{market}-daytrade/trades-YYYY-MM-DD.json
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger('trade_recorder')

TW_TZ = timezone(timedelta(hours=8))


def _now_str() -> str:
    return datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S')


def _today_str() -> str:
    return datetime.now(TW_TZ).strftime('%Y-%m-%d')


class TradeRecorder:
    """
    Records trades to disk immediately on entry.

    Args:
        market: 'tw' or 'us'
        data_dir: e.g. /path/to/project/data/tw-daytrade
    """

    def __init__(self, market: str, data_dir: Path):
        self.market = market
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._trades: List[Dict] = []
        self._load_today_trades()

    @property
    def trades_file(self) -> Path:
        return self.data_dir / f'trades-{_today_str()}.json'

    @property
    def positions_file(self) -> Path:
        return self.data_dir / 'positions.json'

    def _load_today_trades(self) -> None:
        """Load existing trades for today (for crash recovery)."""
        if self.trades_file.exists():
            try:
                self._trades = json.loads(self.trades_file.read_text(encoding='utf-8'))
                logger.info(f"載入今日交易記錄: {len(self._trades)} 筆")
            except Exception as e:
                logger.warning(f"載入交易記錄失敗: {e}")
                self._trades = []

    def _save_trades(self) -> None:
        try:
            self.trades_file.write_text(
                json.dumps(self._trades, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"寫入交易記錄失敗: {e}")

    def _save_positions(self, positions: Dict) -> None:
        try:
            data = {
                'updated_at': _now_str(),
                'market': self.market,
                'positions': positions,
            }
            self.positions_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"寫入 positions.json 失敗: {e}")

    def on_entry(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        score: int,
        signals: List[str],
        stop_price: float,
        take_profit: float,
        side: str = 'long',
    ) -> str:
        """Record trade entry immediately. Returns trade_id."""
        trade_id = f"{symbol}_{int(time.time())}"
        trade = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'status': 'open',
            'entry_time': _now_str(),
            'entry_price': entry_price,
            'qty': qty,
            'score': score,
            'signals': signals,
            'stop_price': stop_price,
            'take_profit': take_profit,
            'exit_time': None,
            'exit_price': None,
            'pnl': None,
            'pnl_pct': None,
            'exit_reason': None,
        }
        self._trades.append(trade)
        self._save_trades()  # IMMEDIATE write
        logger.info(f"[RECORDER] 進場記錄: {symbol} @ {entry_price}")
        return trade_id

    def on_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        reason: str,
    ) -> Optional[Dict]:
        """Update trade record with exit info."""
        for trade in self._trades:
            if trade['trade_id'] == trade_id:
                trade['status'] = 'closed'
                trade['exit_time'] = _now_str()
                trade['exit_price'] = exit_price
                trade['pnl'] = pnl
                if trade['entry_price'] and trade['entry_price'] > 0:
                    trade['pnl_pct'] = round(
                        (exit_price - trade['entry_price']) / trade['entry_price'] * 100, 2
                    )
                trade['exit_reason'] = reason
                self._save_trades()
                logger.info(f"[RECORDER] 出場記錄: {trade['symbol']} PnL={pnl:+,.0f} {reason}")
                return trade
        logger.warning(f"[RECORDER] 找不到 trade_id: {trade_id}")
        return None

    def update_positions(self, positions: Dict) -> None:
        """Write current open positions to positions.json."""
        self._save_positions(positions)

    def get_today_trades(self) -> List[Dict]:
        return list(self._trades)

    def get_today_pnl(self) -> float:
        return sum(t.get('pnl', 0) or 0 for t in self._trades if t['status'] == 'closed')

    def get_open_trades(self) -> List[Dict]:
        return [t for t in self._trades if t['status'] == 'open']

    def load_positions(self) -> Dict:
        """Load positions from positions.json (for crash recovery)."""
        if not self.positions_file.exists():
            return {}
        try:
            data = json.loads(self.positions_file.read_text(encoding='utf-8'))
            return data.get('positions', {})
        except Exception as e:
            logger.warning(f"載入 positions.json 失敗: {e}")
            return {}

    def print_summary(self) -> None:
        """Print today's trade summary to stdout."""
        trades = self.get_today_trades()
        closed = [t for t in trades if t['status'] == 'closed']
        open_trades = [t for t in trades if t['status'] == 'open']
        total_pnl = self.get_today_pnl()
        wins = sum(1 for t in closed if (t.get('pnl') or 0) >= 0)
        losses = len(closed) - wins

        print(f"\n{'='*50}")
        print(f"  {self.market.upper()} 今日交易記錄 ({_today_str()})")
        print(f"{'='*50}")
        print(f"  完成交易: {len(closed)} 筆 ({wins}W / {losses}L)")
        if open_trades:
            print(f"  持倉中: {len(open_trades)} 筆")
        print(f"  日PnL: {total_pnl:+,.0f}")
        print(f"{'='*50}")
        for t in trades:
            status = "✅" if t.get('pnl', 0) and t['pnl'] >= 0 else ("❌" if t.get('pnl', 0) and t['pnl'] < 0 else "⏳")
            pnl_str = f"PnL={t['pnl']:+,.0f}" if t.get('pnl') is not None else "持倉中"
            print(f"  {status} {t['symbol']} {t['entry_time']} | Score={t['score']} | {pnl_str}")
        print()
