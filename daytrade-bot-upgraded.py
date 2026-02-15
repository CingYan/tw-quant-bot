#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股當沖模擬交易 Bot - 升級版
18個技術指標 + 評分系統 + 動態ATR停損停利
"""

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

# ============================================================================
# 依賴檢查和安裝
# ============================================================================
def check_and_install_dependencies():
    """檢查並安裝所需的依賴"""
    required_packages = {
        'yfinance': 'yfinance',
        'pandas': 'pandas',
        'ta': 'ta',
        'numpy': 'numpy'
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
    'SCAN_INTERVAL': 30,       # 無持倉時：每30秒掃描一次（找買入機會）
    'MONITOR_INTERVAL': 5,     # 有持倉時：每5秒監控一次（停損停利）
    'VOLUME_MULTIPLIER': 1.5,  # 成交量倍數
    'BUY_SCORE_THRESHOLD': 60, # 買入評分門檻 (滿分100)
}

# 評分系統權重配置
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
def calculate_buy_score(indicators: Dict) -> Tuple[int, List[str]]:
    """計算買入評分"""
    score = 0
    signals = []
    
    try:
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
        
        return score, signals
    except Exception as e:
        logger.error(f"計算買入評分失敗: {e}")
        return 0, []

def generate_buy_signal(indicators: Dict) -> Tuple[bool, str, int, str]:
    """根據評分制生成買入訊號"""
    score, signals = calculate_buy_score(indicators)
    signal_str = " + ".join(signals) if signals else "無訊號"
    
    if score >= CONFIG['BUY_SCORE_THRESHOLD']:
        return True, signal_str, score, f"總分{score}/100"
    else:
        return False, signal_str, score, f"總分{score}/100 (需{CONFIG['BUY_SCORE_THRESHOLD']})"

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
                    
                    # 寫入警報
                    alert_msg = f"【賣出】{stock} @ {current_price:.2f} | {reason} | 損益: {profit_pct:+.2f}%"
                    write_alert(alert_msg)
                    
                    # 移除持倉
                    del self.positions[stock]
                    save_positions(self.positions)
                    logger.info(f"【賣出】{stock} @ {current_price:.2f} - {reason}")
            else:
                # 檢查買入訊號（只在可開倉時段）
                if self.can_open_position():
                    should_buy, signal_str, score, score_msg = generate_buy_signal(indicators)
                    
                    if should_buy:
                        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # 新增持倉
                        self.positions[stock] = {
                            'entry_price': current_price,
                            'entry_time': entry_time,
                            'signal': signal_str,
                            'score': score
                        }
                        save_positions(self.positions)
                        
                        # 寫入警報
                        alert_msg = f"【買入】{stock} @ {current_price:.2f} | {signal_str} | {score_msg}"
                        write_alert(alert_msg)
                        
                        logger.info(f"【買入】{stock} @ {current_price:.2f} - {signal_str} ({score_msg})")
                    else:
                        logger.debug(f"{stock} 無買入信號 ({score_msg})")
        
        except Exception as e:
            logger.error(f"處理 {stock} 時發生錯誤: {e}")
            logger.debug(traceback.format_exc())
    
    def run(self):
        """主運行迴圈"""
        logger.info("=" * 80)
        logger.info("台股當沖模擬交易 Bot - 18指標評分系統版本")
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
