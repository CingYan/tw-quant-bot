#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMC 結構分析模組 — daytrade-bot v3.0 擴充
提供 BOS / Sweep / CHoCH / FVG / Order Block / POI 偵測
階梯止盈 / 移動停利 / Kelly Criterion / 日損控制

⚠️ 模擬交易專用 — Shioaji API 僅用於資料查詢，禁止真實下單
"""

__version__ = "3.0.0"

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime, time as dt_time
from enum import Enum

# ============================================================================
# 資料結構
# ============================================================================

class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'high' or 'low'
    timestamp: Optional[datetime] = None

@dataclass
class FVG:
    type: str           # 'bullish' or 'bearish'
    top: float
    bottom: float
    ce: float           # Consequent Encroachment (50%)
    index: int          # 形成位置
    mitigated: bool = False

@dataclass
class OrderBlock:
    type: str           # 'bullish' or 'bearish'
    high: float
    low: float
    index: int
    mitigated: bool = False

@dataclass
class StructureEvent:
    type: str           # 'bos', 'choch', 'sweep'
    direction: str      # 'bullish' or 'bearish'
    index: int
    price: float
    swing_ref: float    # 被突破的波段高低點

@dataclass
class POI:
    """Point of Interest — 高概率入場區域"""
    type: str           # 'ob', 'fvg', 'trinity'
    direction: str
    high: float
    low: float
    score: int = 0      # 加權分數
    has_displacement: bool = False
    has_imbalance: bool = False
    has_liquidity_sweep: bool = False

# ============================================================================
# 波段高低點辨識
# ============================================================================

def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> List[SwingPoint]:
    """辨識波段高低點（fractal method）"""
    swings = []
    highs = df['High'].values
    lows = df['Low'].values
    
    for i in range(lookback, len(df) - lookback):
        # Swing High
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(index=i, price=highs[i], type='high'))
        # Swing Low
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(index=i, price=lows[i], type='low'))
    
    return sorted(swings, key=lambda s: s.index)

# ============================================================================
# BOS / CHoCH / Sweep 偵測
# ============================================================================

def detect_structure(df: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureEvent]:
    """
    偵測 BOS、CHoCH、Sweep
    
    規則：
    - BOS：實體收盤突破前波段高/低 → 趨勢延續
    - Sweep：影線刺穿但收盤回到結構內 → 掃止損
    - CHoCH：逆勢突破保護性結構點 → 趨勢反轉
    """
    events = []
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    
    # 追蹤趨勢狀態
    trend = Bias.NEUTRAL
    swing_highs = [s for s in swings if s.type == 'high']
    swing_lows = [s for s in swings if s.type == 'low']
    
    for i in range(1, len(df)):
        # 檢查每個波段高點
        for sh in swing_highs:
            if sh.index >= i:
                continue
            
            # 影線突破但收盤回到結構內 = Sweep
            if highs[i] > sh.price and closes[i] < sh.price:
                events.append(StructureEvent(
                    type='sweep', direction='bearish',
                    index=i, price=highs[i], swing_ref=sh.price
                ))
            
            # 實體收盤突破 = BOS 或 CHoCH
            elif closes[i] > sh.price:
                if trend == Bias.BEARISH:
                    # 逆勢突破 = CHoCH
                    events.append(StructureEvent(
                        type='choch', direction='bullish',
                        index=i, price=closes[i], swing_ref=sh.price
                    ))
                    trend = Bias.BULLISH
                else:
                    # 順勢突破 = BOS
                    events.append(StructureEvent(
                        type='bos', direction='bullish',
                        index=i, price=closes[i], swing_ref=sh.price
                    ))
                    trend = Bias.BULLISH
        
        # 檢查每個波段低點
        for sl in swing_lows:
            if sl.index >= i:
                continue
            
            if lows[i] < sl.price and closes[i] > sl.price:
                events.append(StructureEvent(
                    type='sweep', direction='bullish',
                    index=i, price=lows[i], swing_ref=sl.price
                ))
            elif closes[i] < sl.price:
                if trend == Bias.BULLISH:
                    events.append(StructureEvent(
                        type='choch', direction='bearish',
                        index=i, price=closes[i], swing_ref=sl.price
                    ))
                    trend = Bias.BEARISH
                else:
                    events.append(StructureEvent(
                        type='bos', direction='bearish',
                        index=i, price=closes[i], swing_ref=sl.price
                    ))
                    trend = Bias.BEARISH
    
    return events

# ============================================================================
# FVG 偵測
# ============================================================================

def detect_fvg(df: pd.DataFrame) -> List[FVG]:
    """偵測 Fair Value Gap（三根K棒影線不重疊）"""
    fvgs = []
    highs = df['High'].values
    lows = df['Low'].values
    
    for i in range(2, len(df)):
        # 多頭 FVG：第三根低點 > 第一根高點
        if lows[i] > highs[i - 2]:
            fvgs.append(FVG(
                type='bullish',
                top=lows[i],
                bottom=highs[i - 2],
                ce=(lows[i] + highs[i - 2]) / 2,
                index=i
            ))
        # 空頭 FVG：第三根高點 < 第一根低點
        if highs[i] < lows[i - 2]:
            fvgs.append(FVG(
                type='bearish',
                top=lows[i - 2],
                bottom=highs[i],
                ce=(lows[i - 2] + highs[i]) / 2,
                index=i
            ))
    
    return fvgs

def check_fvg_mitigation(fvgs: List[FVG], df: pd.DataFrame) -> List[FVG]:
    """檢查 FVG 是否已被回測（mitigated）"""
    closes = df['Close'].values
    for fvg in fvgs:
        for i in range(fvg.index + 1, len(df)):
            if fvg.type == 'bullish' and closes[i] < fvg.bottom:
                fvg.mitigated = True
                break
            elif fvg.type == 'bearish' and closes[i] > fvg.top:
                fvg.mitigated = True
                break
    return fvgs

# ============================================================================
# Order Block 偵測
# ============================================================================

def detect_order_blocks(df: pd.DataFrame, events: List[StructureEvent]) -> List[OrderBlock]:
    """偵測 Order Block（BOS/CHoCH 前的最後反方向 K 棒）"""
    obs = []
    opens = df['Open'].values
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    for event in events:
        if event.type not in ('bos', 'choch'):
            continue
        
        idx = event.index
        # 往回找最後一根反方向 K 棒
        for j in range(idx - 1, max(0, idx - 10), -1):
            if event.direction == 'bullish' and closes[j] < opens[j]:
                # 多頭 OB = BOS 前最後一根陰線
                obs.append(OrderBlock(
                    type='bullish', high=highs[j], low=lows[j], index=j
                ))
                break
            elif event.direction == 'bearish' and closes[j] > opens[j]:
                # 空頭 OB = BOS 前最後一根陽線
                obs.append(OrderBlock(
                    type='bearish', high=highs[j], low=lows[j], index=j
                ))
                break
    
    return obs

# ============================================================================
# POI 三位一體偵測
# ============================================================================

def detect_displacement(df: pd.DataFrame, index: int, lookback: int = 10) -> bool:
    """偵測位移（大實體 K 棒）"""
    body = abs(df['Close'].iloc[index] - df['Open'].iloc[index])
    avg_body = abs(df['Close'].iloc[max(0, index - lookback):index] - 
                   df['Open'].iloc[max(0, index - lookback):index]).mean()
    return body > avg_body * 1.5

def detect_poi_trinity(df: pd.DataFrame) -> List[POI]:
    """
    偵測 POI 三位一體區域
    條件：位移 + 失衡(FVG) + 流動性掃除 同時出現
    """
    swings = find_swing_points(df)
    events = detect_structure(df, swings)
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df, events)
    
    pois = []
    
    for ob in obs:
        poi = POI(
            type='ob', direction=ob.type,
            high=ob.high, low=ob.low, score=0
        )
        
        # 檢查位移
        for event in events:
            if abs(event.index - ob.index) <= 3:
                if detect_displacement(df, event.index):
                    poi.has_displacement = True
                    poi.score += 30
                break
        
        # 檢查失衡（附近有 FVG）
        for fvg in fvgs:
            if abs(fvg.index - ob.index) <= 3 and not fvg.mitigated:
                poi.has_imbalance = True
                poi.score += 30
                break
        
        # 檢查流動性掃除（附近有 Sweep）
        for event in events:
            if event.type == 'sweep' and abs(event.index - ob.index) <= 5:
                poi.has_liquidity_sweep = True
                poi.score += 40
                break
        
        # 三位一體 = 最高評分
        if poi.has_displacement and poi.has_imbalance and poi.has_liquidity_sweep:
            poi.type = 'trinity'
            poi.score = 100
        
        pois.append(poi)
    
    return pois

# ============================================================================
# Kill Zone 判斷
# ============================================================================

def get_kill_zone() -> Optional[str]:
    """判斷當前是否在 Kill Zone"""
    now = datetime.now().time()
    
    if dt_time(9, 0) <= now <= dt_time(10, 30):
        return "opening"   # 開盤 KZ
    elif dt_time(12, 0) <= now <= dt_time(13, 0):
        return "closing"   # 午盤 KZ
    elif dt_time(10, 30) < now < dt_time(12, 0):
        return "dead_zone" # 死區
    return None

# ============================================================================
# 階梯止盈系統
# ============================================================================

@dataclass
class LadderTP:
    """階梯止盈管理器"""
    entry_price: float
    atr: float
    direction: str = 'long'
    remaining_pct: float = 100.0
    stop_loss: float = 0.0
    highest_price: float = 0.0
    tp_levels_hit: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.direction == 'long':
            self.stop_loss = self.entry_price - 2 * self.atr
            self.highest_price = self.entry_price
        else:
            self.stop_loss = self.entry_price + 2 * self.atr
            self.highest_price = self.entry_price
    
    def update(self, current_price: float, swing_target: float = 0, 
               yesterday_hl: float = 0) -> Tuple[float, str]:
        """
        更新止盈狀態
        Returns: (平倉比例, 原因)
        """
        if self.direction == 'long':
            self.highest_price = max(self.highest_price, current_price)
        
        # TP1: 1.5×ATR → 平 50%
        tp1 = self.entry_price + 1.5 * self.atr
        if current_price >= tp1 and 'tp1' not in self.tp_levels_hit:
            self.tp_levels_hit.append('tp1')
            self.stop_loss = self.entry_price  # 移到保本
            self.remaining_pct -= 50
            return 50.0, f"TP1 ({tp1:.2f}) 平倉50%，止損移保本"
        
        # TP2: 波段目標 → 平 30%
        if swing_target > 0 and current_price >= swing_target and 'tp2' not in self.tp_levels_hit:
            self.tp_levels_hit.append('tp2')
            self.remaining_pct -= 30
            return 30.0, f"TP2 波段目標 ({swing_target:.2f}) 平倉30%"
        
        # TP3: 昨高/昨低 → 平剩餘
        if yesterday_hl > 0 and current_price >= yesterday_hl and 'tp3' not in self.tp_levels_hit:
            self.tp_levels_hit.append('tp3')
            close_pct = self.remaining_pct
            self.remaining_pct = 0
            return close_pct, f"TP3 昨高 ({yesterday_hl:.2f}) 全部平倉"
        
        # 移動停利（ATR trailing）
        trailing = self.highest_price - 1.5 * self.atr
        if 'tp1' in self.tp_levels_hit:
            self.stop_loss = max(self.stop_loss, trailing)
        
        # 檢查止損
        if current_price <= self.stop_loss:
            close_pct = self.remaining_pct
            self.remaining_pct = 0
            return close_pct, f"止損 ({self.stop_loss:.2f})"
        
        return 0.0, "持倉中"

# ============================================================================
# 日損控制
# ============================================================================

class DailyRiskManager:
    """日內風控管理器"""
    
    def __init__(self, account_balance: float, max_daily_loss_pct: float = 0.03,
                 max_single_risk_pct: float = 0.01, max_consecutive_losses: int = 3):
        self.account_balance = account_balance
        self.max_daily_loss = account_balance * max_daily_loss_pct
        self.max_single_risk = account_balance * max_single_risk_pct
        self.max_consecutive_losses = max_consecutive_losses
        
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.consecutive_losses = 0
        self.is_locked = False
        self.lock_reason = ""
    
    def record_trade(self, pnl: float):
        """記錄交易結果"""
        self.daily_pnl += pnl
        self.trade_count += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self._check_limits()
    
    def _check_limits(self):
        """檢查風控限制"""
        if self.daily_pnl <= -self.max_daily_loss:
            self.is_locked = True
            self.lock_reason = f"日損上限 ({self.daily_pnl:.0f}/{-self.max_daily_loss:.0f})"
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_locked = True
            self.lock_reason = f"連虧 {self.consecutive_losses} 筆，冷靜 30 分鐘"
    
    def can_trade(self) -> Tuple[bool, str]:
        """是否可以交易"""
        if self.is_locked:
            return False, self.lock_reason
        return True, "OK"
    
    def calculate_position_size(self, entry_price: float, stop_price: float) -> int:
        """計算部位大小（股數）"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        shares = int(self.max_single_risk / risk_per_share)
        # 整張（1000股）
        lots = shares // 1000
        return max(0, lots * 1000)

# ============================================================================
# Kelly Criterion
# ============================================================================

def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float, 
                    use_half: bool = True) -> float:
    """
    計算 Kelly 最佳資金比例
    
    Args:
        win_rate: 勝率 (0-1)
        avg_win: 平均獲利金額
        avg_loss: 平均虧損金額
        use_half: 使用 Half Kelly（推薦）
    
    Returns:
        最佳風險比例 (0-0.02，上限 2%)
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.01  # 預設 1%
    
    R = avg_win / avg_loss
    f = win_rate - (1 - win_rate) / R
    
    if use_half:
        f = f / 2
    
    # 上限 2%，下限 0.5%
    return max(0.005, min(0.02, f))

# ============================================================================
# SMC 評分整合（供 daytrade-bot.py 調用）
# ============================================================================

def calculate_smc_score(df_5m: pd.DataFrame, df_15m: pd.DataFrame = None,
                        df_1h: pd.DataFrame = None) -> Dict:
    """
    計算 SMC 結構評分
    
    Args:
        df_5m: 5分鐘 K 線
        df_15m: 15分鐘 K 線（可選）
        df_1h: 1小時 K 線（可選）
    
    Returns:
        {
            'score': int,           # SMC 總分 (0-100)
            'bias': str,            # 方向偏見
            'signals': list,        # 觸發的信號列表
            'pois': list,           # 有效 POI 列表
            'fvgs': list,           # 未回測的 FVG
            'latest_event': str,    # 最新結構事件
        }
    """
    result = {
        'score': 0, 'bias': 'neutral', 'signals': [],
        'pois': [], 'fvgs': [], 'latest_event': ''
    }
    
    # 1H 方向判斷
    htf_bias = Bias.NEUTRAL
    if df_1h is not None and len(df_1h) >= 10:
        swings_1h = find_swing_points(df_1h, lookback=3)
        events_1h = detect_structure(df_1h, swings_1h)
        if events_1h:
            latest = events_1h[-1]
            if latest.type in ('bos', 'choch'):
                htf_bias = Bias.BULLISH if latest.direction == 'bullish' else Bias.BEARISH
                result['bias'] = htf_bias.value
                result['score'] += 15
                result['signals'].append(f"1H {latest.type.upper()} {latest.direction}")
    
    # 15M POI
    if df_15m is not None and len(df_15m) >= 20:
        pois_15m = detect_poi_trinity(df_15m)
        active_pois = [p for p in pois_15m if p.score >= 30]
        result['pois'] = active_pois
        if active_pois:
            best_poi = max(active_pois, key=lambda p: p.score)
            result['score'] += min(25, best_poi.score // 4)
            result['signals'].append(f"15M POI({best_poi.type}, score={best_poi.score})")
    
    # 5M 結構分析
    if len(df_5m) >= 20:
        swings_5m = find_swing_points(df_5m, lookback=3)
        events_5m = detect_structure(df_5m, swings_5m)
        fvgs_5m = detect_fvg(df_5m)
        fvgs_5m = check_fvg_mitigation(fvgs_5m, df_5m)
        
        active_fvgs = [f for f in fvgs_5m if not f.mitigated]
        result['fvgs'] = active_fvgs
        
        if events_5m:
            latest = events_5m[-1]
            result['latest_event'] = f"{latest.type}_{latest.direction}"
            
            # CHoCH 加分
            if latest.type == 'choch':
                result['score'] += 20
                result['signals'].append(f"5M CHoCH {latest.direction}")
            
            # Sweep + CHoCH 組合加分
            sweeps = [e for e in events_5m if e.type == 'sweep']
            chochs = [e for e in events_5m if e.type == 'choch']
            if sweeps and chochs:
                last_sweep = sweeps[-1]
                last_choch = chochs[-1]
                if last_choch.index > last_sweep.index and last_choch.index - last_sweep.index <= 10:
                    result['score'] += 15
                    result['signals'].append("Sweep→CHoCH 組合")
        
        # FVG 在 CHoCH 後 = 入場區域
        if active_fvgs and events_5m:
            for fvg in active_fvgs[-3:]:  # 最近 3 個
                for event in events_5m[-5:]:
                    if event.type == 'choch' and fvg.index > event.index:
                        result['score'] += 10
                        result['signals'].append(f"CHoCH後FVG({fvg.type})")
                        break
    
    # Kill Zone 加分
    kz = get_kill_zone()
    if kz == 'opening':
        result['score'] += 10
        result['signals'].append("開盤KZ")
    elif kz == 'closing':
        result['score'] += 5
        result['signals'].append("午盤KZ")
    elif kz == 'dead_zone':
        result['score'] -= 10
        result['signals'].append("⚠️死區")
    
    # 方向一致加分
    if htf_bias != Bias.NEUTRAL and result.get('latest_event', ''):
        if htf_bias.value in result['latest_event']:
            result['score'] += 10
            result['signals'].append("多時框方向一致")
    
    return result

# ============================================================================
# 入口：測試用
# ============================================================================

if __name__ == '__main__':
    print(f"SMC 結構分析模組 v{__version__}")
    print("⚠️ 模擬交易專用")
    print()
    print("可用函數：")
    print("  find_swing_points(df)        - 辨識波段高低點")
    print("  detect_structure(df, swings) - 偵測 BOS/CHoCH/Sweep")
    print("  detect_fvg(df)               - 偵測 Fair Value Gap")
    print("  detect_order_blocks(df, ev)  - 偵測 Order Block")
    print("  detect_poi_trinity(df)       - 偵測 POI 三位一體")
    print("  calculate_smc_score(...)     - 計算 SMC 綜合評分")
    print("  LadderTP(...)                - 階梯止盈管理器")
    print("  DailyRiskManager(...)        - 日損控制管理器")
    print("  kelly_criterion(...)         - Kelly Criterion 計算")
