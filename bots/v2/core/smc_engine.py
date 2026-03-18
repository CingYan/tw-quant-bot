# -*- coding: utf-8 -*-
"""
SMC Engine v2.0 — Shared Smart Money Concepts Analysis
OB / FVG / BOS / CHoCH / Sweep + VWAP multi-timeframe scoring

Key v2 fixes from trading-lessons.md:
- Multi-timeframe conflict → HARD REJECT (not score penalty)
- FVG saturation >= 7 → HARD REJECT
- FVG scoring: saturation-based (not fixed +10)
- VWAP distance filter: reject if price > 1.5 ATR from VWAP
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
    import ta
except ImportError:
    pass  # Caller must ensure deps installed

FVG_SATURATION_THRESHOLD = 7


class Bias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class FVG:
    type: str         # 'bullish' or 'bearish'
    top: float
    bottom: float
    ce: float         # center of gap
    index: int
    mitigated: bool = False


@dataclass
class OrderBlock:
    type: str         # 'bullish' or 'bearish'
    high: float
    low: float
    index: int
    mitigated: bool = False


@dataclass
class StructureEvent:
    type: str         # 'bos', 'choch', 'sweep'
    direction: str    # 'bullish' or 'bearish'
    index: int
    price: float
    swing_ref: float


@dataclass
class SMCResult:
    score: int = 0
    bias: str = 'neutral'
    signals: List[str] = field(default_factory=list)
    entry_zone: Tuple[float, float] = (0.0, 0.0)
    stop_price: float = 0.0
    latest_event: str = ''
    vwap_score: int = 0
    rejected: bool = False
    reject_reason: str = ''

    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'bias': self.bias,
            'signals': self.signals,
            'entry_zone': list(self.entry_zone),
            'stop_price': self.stop_price,
            'latest_event': self.latest_event,
            'vwap_score': self.vwap_score,
            'rejected': self.rejected,
            'reject_reason': self.reject_reason,
        }


# ============================================================================
# Structure detection primitives
# ============================================================================

def find_swing_points(df: 'pd.DataFrame', lookback: int = 5) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    highs = df['High'].values
    lows = df['Low'].values
    for i in range(lookback, len(df) - lookback):
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(index=i, price=float(highs[i]), type='high'))
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            swings.append(SwingPoint(index=i, price=float(lows[i]), type='low'))
    return sorted(swings, key=lambda s: s.index)


def detect_structure(df: 'pd.DataFrame', swings: List[SwingPoint]) -> List[StructureEvent]:
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


def detect_fvg(df: 'pd.DataFrame') -> List[FVG]:
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


def check_fvg_mitigation(fvgs: List[FVG], df: 'pd.DataFrame') -> List[FVG]:
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


def detect_order_blocks(df: 'pd.DataFrame', events: List[StructureEvent]) -> List[OrderBlock]:
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


def detect_displacement(df: 'pd.DataFrame', index: int, lookback: int = 10) -> bool:
    if index < lookback:
        return False
    body = abs(df['Close'].iloc[index] - df['Open'].iloc[index])
    past_bodies = abs(
        df['Close'].iloc[max(0, index - lookback):index] -
        df['Open'].iloc[max(0, index - lookback):index]
    )
    avg_body = past_bodies.mean()
    return bool(body > avg_body * 1.5) if avg_body > 0 else False


def calculate_atr(df: 'pd.DataFrame', period: int = 14) -> float:
    try:
        atr_series = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=period
        )
        val = atr_series.iloc[-1]
        if np.isnan(val) or val <= 0:
            return float((df['High'] - df['Low']).tail(5).mean())
        return float(val)
    except Exception:
        return float(df['Close'].iloc[-1] * 0.02)


# ============================================================================
# VWAP Calculator
# ============================================================================

class VWAPCalculator:

    @staticmethod
    def calculate(df: 'pd.DataFrame') -> Dict:
        if df.empty or len(df) < 5:
            last_price = float(df['Close'].iloc[-1]) if not df.empty else 0.0
            return {
                'vwap': last_price, 'upper_1': last_price, 'lower_1': last_price,
                'upper_2': last_price, 'lower_2': last_price,
                'series_vwap': None,
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
    def detect_bounce(df: 'pd.DataFrame', vwap_data: Dict) -> Dict:
        result = {
            'bounce_long': False, 'rejection_short': False,
            'near_vwap': False, 'bounce_strength': 'none'
        }
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
    def get_score_contribution(df: 'pd.DataFrame', vwap_data: Dict, smc_bias: str,
                                atr: float = 0.0) -> Tuple[int, List[str]]:
        score = 0
        signals: List[str] = []
        if not vwap_data or not vwap_data.get('vwap'):
            return 0, signals
        current_price = float(df['Close'].iloc[-1])
        vwap = vwap_data['vwap']
        # VWAP distance filter: reject if price > 1.5 ATR from VWAP (v2 fix)
        if atr > 0:
            distance = abs(current_price - vwap)
            if distance > atr * 1.5:
                signals.append(f"⛔ VWAP距離過遠({distance/atr:.1f}ATR>1.5)")
                return -999, signals  # Signal hard reject to caller
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
# SMC Engine — main class
# ============================================================================

class SMCEngine:
    """
    Shared SMC analysis engine for both TW and US bots.

    Usage:
        engine = SMCEngine()
        result = engine.analyze(df_5m, df_15m, df_1h, vwap_data, kill_zone)
        if result.rejected:
            return  # hard reject
        if result.score >= min_score:
            enter_trade(result)
    """

    def analyze(
        self,
        df_5m: 'pd.DataFrame',
        df_15m: Optional['pd.DataFrame'],
        df_1h: Optional['pd.DataFrame'],
        vwap_data: Optional[Dict] = None,
        kill_zone: Optional[str] = None,
        is_kz1: bool = False,
        atr: float = 0.0,
    ) -> SMCResult:
        result = SMCResult()

        bias_1h = Bias.NEUTRAL
        bias_15m = Bias.NEUTRAL
        bias_5m = Bias.NEUTRAL

        # --- 1H ---
        if df_1h is not None and len(df_1h) >= 15:
            try:
                swings_1h = find_swing_points(df_1h, lookback=3)
                events_1h = detect_structure(df_1h, swings_1h)
                if events_1h:
                    latest_1h = events_1h[-1]
                    if latest_1h.type in ('bos', 'choch'):
                        bias_1h = Bias.BULLISH if latest_1h.direction == 'bullish' else Bias.BEARISH
                        result.bias = bias_1h.value
                        result.score += 15
                        result.signals.append(f"1H {latest_1h.type.upper()} {latest_1h.direction}")
            except Exception:
                pass

        # --- 15M ---
        if df_15m is not None and len(df_15m) >= 20:
            try:
                swings_15m = find_swing_points(df_15m, lookback=3)
                events_15m = detect_structure(df_15m, swings_15m)
                obs_15m = detect_order_blocks(df_15m, events_15m)
                fvgs_15m_raw = detect_fvg(df_15m)
                fvgs_15m = check_fvg_mitigation(fvgs_15m_raw, df_15m)

                if events_15m:
                    latest_15m = events_15m[-1]
                    if latest_15m.type in ('bos', 'choch'):
                        bias_15m = Bias.BULLISH if latest_15m.direction == 'bullish' else Bias.BEARISH

                if obs_15m:
                    ob = obs_15m[-1]
                    if not ob.mitigated:
                        result.entry_zone = (ob.low, ob.high)
                        result.score += 20
                        result.signals.append(f"15M OB({ob.type})")

                active_fvgs = [f for f in fvgs_15m if not f.mitigated]
                if active_fvgs:
                    fvg_count = len(active_fvgs)
                    # v2 HARD REJECT: FVG saturation >= 7
                    if fvg_count >= FVG_SATURATION_THRESHOLD:
                        result.rejected = True
                        result.reject_reason = f"FVG飽和 x{fvg_count} → 拒絕進場"
                        result.signals.append(f"⛔ FVG飽和 x{fvg_count}")
                        return result
                    # FVG saturation-based scoring (v2 fix)
                    if fvg_count <= 3:
                        fvg_score = 10
                    elif fvg_count <= 5:
                        fvg_score = 7
                    else:
                        fvg_score = 3  # high count, reduced but not zero
                    result.score += fvg_score
                    result.signals.append(f"15M FVG x{fvg_count}(+{fvg_score})")
            except Exception:
                pass

        # --- 5M ---
        if len(df_5m) >= 20:
            try:
                swings_5m = find_swing_points(df_5m, lookback=3)
                events_5m = detect_structure(df_5m, swings_5m)
                fvgs_5m = check_fvg_mitigation(detect_fvg(df_5m), df_5m)

                if events_5m:
                    latest_5m = events_5m[-1]
                    if latest_5m.type in ('bos', 'choch'):
                        bias_5m = Bias.BULLISH if latest_5m.direction == 'bullish' else Bias.BEARISH
                    result.latest_event = f"{latest_5m.type}_{latest_5m.direction}"

                    if latest_5m.type == 'choch':
                        result.score += 20
                        result.signals.append(f"5M CHoCH {latest_5m.direction}")
                    elif latest_5m.type == 'bos':
                        result.score += 10
                        result.signals.append(f"5M BOS {latest_5m.direction}")

                    sweeps = [e for e in events_5m if e.type == 'sweep']
                    chochs = [e for e in events_5m if e.type == 'choch']
                    if sweeps and chochs:
                        last_sweep = sweeps[-1]
                        last_choch = chochs[-1]
                        if last_choch.index > last_sweep.index and last_choch.index - last_sweep.index <= 10:
                            result.score += 15
                            result.signals.append("Sweep→CHoCH")

                active_fvgs_5m = [f for f in fvgs_5m if not f.mitigated]
                if active_fvgs_5m and events_5m:
                    for fvg in active_fvgs_5m[-3:]:
                        for ev in events_5m[-5:]:
                            if ev.type == 'choch' and fvg.index > ev.index:
                                result.score += 10
                                result.signals.append(f"CHoCH後FVG({fvg.type})")
                                if result.entry_zone == (0.0, 0.0):
                                    result.entry_zone = (fvg.bottom, fvg.top)
                                break
            except Exception:
                pass

        # --- Kill Zone bonus ---
        if kill_zone == 'kz1_open':
            result.score += 15
            result.signals.append("🟢 KZ1 開盤")
        elif kill_zone in ('kz2_close', 'kz2'):
            result.score += 8
            result.signals.append("🟠 KZ2")
        elif kill_zone in ('kz3', 'kz3_close'):
            result.score += 8
            result.signals.append("🔵 KZ3")
        else:
            result.score -= 5
            result.signals.append("⚠️ 非KZ時段")

        # --- Multi-timeframe bias alignment / conflict ---
        active_biases = [
            (bias_1h, "1H"),
            (bias_15m, "15M"),
            (bias_5m, "5M"),
        ]
        valid_biases = [(b, tf) for b, tf in active_biases if b != Bias.NEUTRAL]

        if len(valid_biases) >= 2:
            bias_set = set(b.value for b, _ in valid_biases)
            if len(bias_set) == 1:
                result.score += 10
                result.signals.append("✓ 多時框方向一致(+10)")
            else:
                # v2 HARD REJECT: multi-timeframe conflict
                conflicting_tfs = "/".join([tf for b, tf in valid_biases])
                result.rejected = True
                result.reject_reason = f"多時框衝突 ({conflicting_tfs}) → 拒絕進場"
                result.signals.append(f"⛔ 多時框衝突 ({conflicting_tfs})")
                return result

        # --- VWAP ---
        if vwap_data is not None and df_5m is not None and not df_5m.empty:
            try:
                v_score, v_signals = VWAPCalculator.get_score_contribution(
                    df_5m, vwap_data, result.bias, atr=atr
                )
                result.signals.extend(v_signals)
                if v_score == -999:
                    # VWAP distance hard reject
                    result.rejected = True
                    result.reject_reason = "VWAP距離過遠 → 拒絕進場"
                    return result
                result.score += v_score
                result.vwap_score = v_score
            except Exception:
                pass

        result.score = max(0, min(100, result.score))
        return result
