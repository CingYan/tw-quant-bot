#!/usr/bin/env -S uv run --python 3.13 python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.28",
# ]
# ///
# -*- coding: utf-8 -*-
"""
美股當沖 Failsafe 強制平倉腳本 v2.0

完全獨立腳本 — 不依賴主 Bot 是否在線
排程執行時間: 每日 04:50 台灣時間（= 15:50 ET 冬季）

功能:
  1. 讀取 data/us-daytrade/positions.json
  2. 對所有未平倉部位發送 Alpaca 賣出指令
  3. 傳送 Telegram 通知

用法: python3 failsafe-close-us.py
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
from typing import Dict, List, Optional

# ============================================================================
# Paths & Config
# ============================================================================
BASE_DIR = Path('/path/to/project')
CRED_FILE = BASE_DIR / 'credentials' / 'credentials.json'
DATA_DIR = BASE_DIR / 'data' / 'us-daytrade'
POSITIONS_FILE = DATA_DIR / 'positions.json'
OPENCLAW_CONFIG = Path('/path/to/openclaw.json')

TW_TZ = timezone(timedelta(hours=8))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('failsafe_us')

# ============================================================================
# Telegram (standalone)
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
# Alpaca REST (standalone — no dependency on bot code)
# ============================================================================
def _alpaca_request(
    method: str, endpoint: str,
    api_key: str, secret_key: str,
    base_url: str = 'https://paper-api.alpaca.markets',
    body: Optional[Dict] = None,
) -> Optional[Dict]:
    url = f"{base_url}{endpoint}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('APCA-API-KEY-ID', api_key)
    req.add_header('APCA-API-SECRET-KEY', secret_key)
    req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.debug(f"Alpaca {method} {endpoint}: {e}")
        return None


def alpaca_get_positions(api_key: str, secret_key: str, base_url: str) -> List[Dict]:
    result = _alpaca_request('GET', '/v2/positions', api_key, secret_key, base_url)
    return result if isinstance(result, list) else []


def alpaca_close_all_positions(api_key: str, secret_key: str, base_url: str) -> Optional[Dict]:
    """Close ALL positions at once via DELETE /v2/positions."""
    return _alpaca_request('DELETE', '/v2/positions', api_key, secret_key, base_url)


def alpaca_close_position(symbol: str, api_key: str, secret_key: str, base_url: str) -> Optional[Dict]:
    return _alpaca_request('DELETE', f'/v2/positions/{symbol}', api_key, secret_key, base_url)

# ============================================================================
# Main failsafe logic
# ============================================================================
def main():
    now_tw = datetime.now(TW_TZ)
    logger.info(f"=== US Failsafe 強制平倉 {now_tw.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Load credentials
    creds = load_credentials()
    alpaca_creds = creds.get('alpaca', {})
    api_key = alpaca_creds.get('api_key', '')
    secret_key = alpaca_creds.get('secret_key', '')
    base_url = alpaca_creds.get('base_url', 'https://paper-api.alpaca.markets')

    if not api_key or not secret_key:
        logger.error("Alpaca credentials 未設定")
        _tg_notify("🔴 US Failsafe: Alpaca credentials 未設定")
        return

    # Check actual Alpaca positions (not just local positions.json)
    live_positions = alpaca_get_positions(api_key, secret_key, base_url)
    local_positions: Dict = {}

    if POSITIONS_FILE.exists():
        try:
            data = json.loads(POSITIONS_FILE.read_text(encoding='utf-8'))
            local_positions = data.get('positions', {})
        except Exception as e:
            logger.warning(f"讀取 positions.json 失敗: {e}")

    all_symbols = set(list(local_positions.keys()) + [p.get('symbol', '') for p in live_positions])
    all_symbols.discard('')

    if not all_symbols:
        logger.info("無任何持倉，無需強制平倉")
        _tg_notify("⚡ US Failsafe: 無持倉，跳過")
        return

    logger.warning(f"發現 {len(all_symbols)} 個持倉: {all_symbols}")
    _tg_notify(
        f"⚠️ <b>US Failsafe 啟動</b>\n"
        f"Alpaca持倉: {len(live_positions)} 個\n"
        f"本地記錄: {len(local_positions)} 個\n"
        f"開始強制平倉..."
    )

    closed = []
    failed = []

    # Try bulk close first
    if live_positions:
        result = alpaca_close_all_positions(api_key, secret_key, base_url)
        if result is not None:
            logger.info(f"Alpaca 全部平倉指令已送出: {result}")
            closed = [p.get('symbol', '') for p in live_positions]
        else:
            # Fallback: close one by one
            for pos in live_positions:
                symbol = pos.get('symbol', '')
                if not symbol:
                    continue
                r = alpaca_close_position(symbol, api_key, secret_key, base_url)
                if r is not None:
                    closed.append(symbol)
                    logger.info(f"Failsafe 平倉 {symbol}: OK")
                else:
                    failed.append(symbol)
                    logger.error(f"Failsafe 平倉 {symbol}: 失敗")

    # Also try any local-only positions not in Alpaca
    local_only = set(local_positions.keys()) - set(p.get('symbol', '') for p in live_positions)
    for symbol in local_only:
        logger.info(f"本地記錄有但 Alpaca 無: {symbol} (可能已平倉)")

    # Update local positions file
    try:
        remaining = {s: p for s, p in local_positions.items() if s not in closed}
        updated_data = {
            'updated_at': now_tw.isoformat(),
            'market': 'us',
            'positions': remaining,
            'failsafe_closed': closed,
        }
        POSITIONS_FILE.write_text(
            json.dumps(updated_data, ensure_ascii=False, indent=2)
        )
    except Exception as e:
        logger.error(f"更新 positions.json 失敗: {e}")

    summary = f"✅ 平倉: {', '.join(closed)}" if closed else "無成功平倉"
    fail_str = f"\n❌ 失敗: {', '.join(failed)}" if failed else ""
    _tg_notify(
        f"🏁 <b>US Failsafe 完成</b>\n{summary}{fail_str}\n"
        f"時間: {now_tw.strftime('%H:%M:%S')}"
    )
    logger.info(f"US Failsafe 完成: closed={closed}, failed={failed}")


if __name__ == '__main__':
    main()
