#!/usr/bin/env -S uv run --python 3.13 python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "shioaji>=1.3",
#   "requests>=2.28",
# ]
# ///
# -*- coding: utf-8 -*-
"""
台股當沖 Failsafe 強制平倉腳本 v2.0

完全獨立腳本 — 不依賴主 Bot 是否在線
排程執行時間: 每日 13:20 台灣時間

功能:
  1. 讀取 data/tw-daytrade/positions.json
  2. 對所有未平倉部位發送賣出指令
  3. 傳送 Telegram 通知

用法: python3 failsafe-close-tw.py
"""

import sys
import os
import json
import urllib.parse
import urllib.request
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# Paths & Config
# ============================================================================
BASE_DIR = Path('/path/to/project')
CRED_FILE = BASE_DIR / 'credentials' / 'credentials.json'
DATA_DIR = BASE_DIR / 'data' / 'tw-daytrade'
POSITIONS_FILE = DATA_DIR / 'positions.json'
OPENCLAW_CONFIG = Path('/path/to/openclaw.json')

TW_TZ = timezone(timedelta(hours=8))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('failsafe_tw')

# ============================================================================
# Telegram (standalone, no core import)
# ============================================================================
def _tg_notify(text: str, chat_id: str = 'YOUR_TELEGRAM_CHAT_ID') -> None:
    try:
        config = json.loads(OPENCLAW_CONFIG.read_text())
        channels = config.get('channels', {})
        token = None
        if isinstance(channels, dict):
            tg = channels.get('telegram', {})
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
        }).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        logger.debug(f"Telegram 通知失敗: {e}")

# ============================================================================
# Credentials
# ============================================================================
def load_credentials() -> Dict:
    try:
        return json.loads(CRED_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"credentials 載入失敗: {e}")
        sys.exit(1)

# ============================================================================
# Shioaji (standalone)
# ============================================================================
def get_shioaji_api(api_key: str, secret_key: str):
    try:
        import shioaji as sj
        api = sj.Shioaji(simulation=True)
        api.login(
            api_key=api_key,
            secret_key=secret_key,
            contracts_cb=lambda t: None,
        )
        logger.info("Failsafe: Shioaji 登入成功")
        return api
    except Exception as e:
        logger.error(f"Failsafe: Shioaji 登入失敗: {e}")
        return None


def shioaji_sell_all(api, symbol: str, qty_shares: int) -> bool:
    """Sell all shares of a symbol."""
    lots = qty_shares // 1000
    if lots <= 0:
        logger.warning(f"{symbol}: qty={qty_shares} 不足 1 張，跳過")
        return False
    try:
        import shioaji as sj
        contract = api.Contracts.Stocks.get(symbol) or api.Contracts.Stocks[symbol]
        if not contract:
            logger.error(f"{symbol}: 找不到合約")
            return False
        order = api.Order(
            price=0,
            quantity=lots,
            action=sj.constant.Action.Sell,
            price_type=sj.constant.StockPriceType.MKT,
            order_type=sj.constant.OrderType.IOC,
            order_lot=sj.constant.StockOrderLot.Common,
            account=api.stock_account,
        )
        trade = api.place_order(contract, order)
        time.sleep(1)
        api.update_status(api.stock_account)
        status = str(getattr(trade.status, 'status', ''))
        logger.info(f"Failsafe 賣出 {symbol} {lots}張: {status}")
        return True
    except Exception as e:
        logger.error(f"Failsafe {symbol} 賣出失敗: {e}")
        return False

# ============================================================================
# Main failsafe logic
# ============================================================================
def main():
    now_tw = datetime.now(TW_TZ)
    logger.info(f"=== TW Failsafe 強制平倉 {now_tw.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Load positions
    if not POSITIONS_FILE.exists():
        logger.info("positions.json 不存在，無需操作")
        _tg_notify("⚡ TW Failsafe: 無持倉，跳過")
        return

    try:
        data = json.loads(POSITIONS_FILE.read_text(encoding='utf-8'))
        positions = data.get('positions', {})
    except Exception as e:
        logger.error(f"讀取 positions.json 失敗: {e}")
        _tg_notify(f"🔴 TW Failsafe: 讀取持倉失敗 {e}")
        return

    if not positions:
        logger.info("持倉為空，無需強制平倉")
        _tg_notify("⚡ TW Failsafe: 無持倉，跳過")
        return

    logger.warning(f"發現 {len(positions)} 個持倉，開始強制平倉...")
    _tg_notify(f"⚠️ <b>TW Failsafe 啟動</b>\n發現 {len(positions)} 個持倉\n開始強制平倉...")

    # Load credentials and connect
    creds = load_credentials()
    shioaji_creds = creds.get('shioaji', {})
    api_key = shioaji_creds.get('api_key', '')
    secret_key = shioaji_creds.get('secret_key', '')

    if not api_key or not secret_key:
        logger.error("Shioaji credentials 未設定")
        _tg_notify("🔴 TW Failsafe: Shioaji credentials 未設定")
        return

    api = get_shioaji_api(api_key, secret_key)
    if api is None:
        _tg_notify("🔴 TW Failsafe: Shioaji 登入失敗，無法平倉")
        return

    closed = []
    failed = []

    for symbol, pos in positions.items():
        qty = pos.get('qty', 0)
        entry_price = pos.get('entry_price', 0)
        logger.info(f"強制賣出: {symbol} qty={qty} entry={entry_price}")
        success = shioaji_sell_all(api, symbol, qty)
        if success:
            closed.append(symbol)
        else:
            failed.append(symbol)

    # Clear positions file
    if closed:
        try:
            remaining = {s: p for s, p in positions.items() if s not in closed}
            updated_data = {
                'updated_at': now_tw.isoformat(),
                'market': 'tw',
                'positions': remaining,
                'failsafe_closed': closed,
            }
            POSITIONS_FILE.write_text(
                json.dumps(updated_data, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.error(f"更新 positions.json 失敗: {e}")

    # Logout
    try:
        api.logout()
    except Exception:
        pass

    # Summary notification
    summary = f"✅ 平倉: {', '.join(closed)}" if closed else "無成功平倉"
    fail_str = f"\n❌ 失敗: {', '.join(failed)}" if failed else ""
    _tg_notify(
        f"🏁 <b>TW Failsafe 完成</b>\n{summary}{fail_str}\n"
        f"時間: {now_tw.strftime('%H:%M:%S')}"
    )
    logger.info(f"Failsafe 完成: closed={closed}, failed={failed}")


if __name__ == '__main__':
    main()
