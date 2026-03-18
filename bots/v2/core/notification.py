# -*- coding: utf-8 -*-
"""
Telegram Notifier v2.0 — Shared notification module

Reads bot token from /path/to/openclaw.json
Silently fails if Telegram is unavailable.
"""

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger('notification')

OPENCLAW_CONFIG = Path('/path/to/openclaw.json')


def _load_bot_token() -> Optional[str]:
    try:
        config = json.loads(OPENCLAW_CONFIG.read_text())
        channels = config.get('channels', {})
        if isinstance(channels, dict):
            tg = channels.get('telegram', {})
            if isinstance(tg, dict):
                return tg.get('botToken') or tg.get('token')
        elif isinstance(channels, list):
            for ch in channels:
                if isinstance(ch, dict) and ch.get('type') == 'telegram':
                    return ch.get('botToken') or ch.get('token')
    except Exception:
        pass
    return None


class TelegramNotifier:
    """
    Sends HTML-formatted Telegram messages.
    All methods silently fail on error.
    """

    def __init__(self, chat_id: str = 'YOUR_TELEGRAM_CHAT_ID'):
        self.chat_id = chat_id
        self.bot_token = _load_bot_token()
        if not self.bot_token:
            logger.warning("Telegram bot token 未設定，通知功能停用")

    def notify(self, text: str) -> bool:
        """Send raw HTML message. Returns True on success."""
        if not self.bot_token:
            return False
        try:
            url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
            data = urllib.parse.urlencode({
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
                'disable_notification': 'false',
            }).encode()
            urllib.request.urlopen(
                urllib.request.Request(url, data=data), timeout=10
            )
            return True
        except Exception as e:
            logger.debug(f"Telegram 通知失敗: {e}")
            return False

    def on_startup(self, market: str, watchlist_count: int, capital: float) -> None:
        msg = (
            f"🚀 <b>{market} Bot v2.0 啟動</b>\n"
            f"📋 監控標的: {watchlist_count} 支\n"
            f"💰 資金: {capital:,.0f}\n"
            f"⚡ 自動化交易已開始"
        )
        self.notify(msg)

    def on_entry(
        self, symbol: str, price: float, qty: float,
        score: int, signals: list, stop: float, tp: float,
        market: str = ''
    ) -> None:
        sig_str = ' | '.join(signals[:4]) if signals else '-'
        market_tag = f"[{market}] " if market else ""
        msg = (
            f"📈 <b>{market_tag}進場: {symbol}</b>\n"
            f"💵 價格: {price:.2f} × {qty}\n"
            f"🎯 SMC Score: {score}\n"
            f"📊 信號: {sig_str}\n"
            f"🛑 停損: {stop:.2f}\n"
            f"✅ 目標: {tp:.2f}"
        )
        self.notify(msg)

    def on_exit(
        self, symbol: str, entry_price: float, exit_price: float,
        qty: float, pnl: float, reason: str, market: str = ''
    ) -> None:
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
        market_tag = f"[{market}] " if market else ""
        msg = (
            f"{pnl_emoji} <b>{market_tag}出場: {symbol}</b>\n"
            f"💵 {entry_price:.2f} → {exit_price:.2f} ({pct:+.2f}%)\n"
            f"📦 數量: {qty}\n"
            f"💰 PnL: {pnl:+,.0f}\n"
            f"📌 原因: {reason}"
        )
        self.notify(msg)

    def on_shutdown(self, market: str, trades_today: int, daily_pnl: float) -> None:
        pnl_emoji = "✅" if daily_pnl >= 0 else "❌"
        msg = (
            f"🏁 <b>{market} Bot 收盤</b>\n"
            f"📊 今日交易: {trades_today} 筆\n"
            f"{pnl_emoji} 日PnL: {daily_pnl:+,.0f}\n"
            f"💤 Bot 已停止"
        )
        self.notify(msg)

    def on_force_close(self, market: str, symbol: str, reason: str = "強制平倉") -> None:
        msg = (
            f"⚠️ <b>{market} 強制平倉: {symbol}</b>\n"
            f"📌 原因: {reason}"
        )
        self.notify(msg)

    def on_day_stop(self, market: str, reason: str) -> None:
        msg = f"⛔ <b>{market} 今日停止交易</b>\n📌 {reason}"
        self.notify(msg)

    def on_error(self, error_msg: str, market: str = '') -> None:
        tag = f"[{market}] " if market else ""
        msg = f"🔴 <b>{tag}Bot 錯誤</b>\n<code>{error_msg[:300]}</code>"
        self.notify(msg)

    def on_restart(self, market: str, reason: str, attempt: int) -> None:
        msg = f"🔄 <b>{market} Bot 重啟</b> (第{attempt}次)\n📌 {reason}"
        self.notify(msg)

    def on_heartbeat_alert(self, market: str, last_hb_sec: int) -> None:
        msg = f"⚠️ <b>{market} Heartbeat 異常</b>\n最後心跳: {last_hb_sec}s 前"
        self.notify(msg)
