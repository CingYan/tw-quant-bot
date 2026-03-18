# -*- coding: utf-8 -*-
"""
Risk Manager v2.0 — Shared position sizing and daily risk control

Key v2 changes:
- Consecutive 3 losses → day_stopped = True (no auto-unlock during same day)
- Per-symbol stop-loss ban (no re-entry same symbol same day after stop)
- Re-entry cooldown: 600s per symbol after stop-loss
- Daily loss limit: hard stop
"""

import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger('risk_manager')


class RiskManager:
    """
    Shared risk manager for TW and US bots.

    Args:
        capital: Starting capital (TWD for TW, USD for US)
        max_daily_loss_pct: e.g. 0.015 = 1.5%
        max_positions: Max concurrent positions (default 3)
        kelly_fraction: Kelly fraction multiplier (default 0.3)
        max_single_risk_pct: Max risk per trade as % of capital (default 0.005)
        consecutive_loss_limit: Losses before day stop (default 3)
        re_entry_cooldown_sec: Per-symbol cooldown after stop-loss (default 600)
    """

    def __init__(
        self,
        capital: float,
        max_daily_loss_pct: float = 0.015,
        max_positions: int = 3,
        kelly_fraction: float = 0.3,
        max_single_risk_pct: float = 0.005,
        consecutive_loss_limit: int = 3,
        re_entry_cooldown_sec: int = 600,
    ):
        self.capital = capital
        self.current_balance = capital
        self.max_daily_loss = capital * max_daily_loss_pct
        self.max_positions = max_positions
        self.kelly_fraction = kelly_fraction
        self.max_single_risk_pct = max_single_risk_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.re_entry_cooldown_sec = re_entry_cooldown_sec

        # State
        self.daily_pnl: float = 0.0
        self.trade_count: int = 0
        self.consecutive_losses: int = 0
        self.day_stopped: bool = False
        self.day_stop_reason: str = ''

        # Per-symbol bans (stop-loss ban)
        self._symbol_stop_ban: Dict[str, float] = {}    # symbol → ban_until timestamp
        self._symbol_cooldown: Dict[str, float] = {}    # symbol → cooldown_until timestamp

        logger.info(
            f"RiskManager: 資金={capital:,.0f}, 日損上限={self.max_daily_loss:,.0f}, "
            f"最大持倉={max_positions}"
        )

    # -------------------------------------------------------------------------
    # Trade gate
    # -------------------------------------------------------------------------

    def can_trade(self, symbol: Optional[str] = None) -> Tuple[bool, str]:
        """Returns (can_trade, reason)"""
        if self.day_stopped:
            return False, self.day_stop_reason

        if self.daily_pnl <= -self.max_daily_loss:
            self.day_stopped = True
            self.day_stop_reason = f"日損上限 {self.daily_pnl:,.0f}"
            return False, self.day_stop_reason

        if symbol:
            # Check stop-loss ban
            ban_until = self._symbol_stop_ban.get(symbol, 0)
            if time.time() < ban_until:
                return False, f"{symbol} 停損ban中"

            # Check re-entry cooldown
            cooldown_until = self._symbol_cooldown.get(symbol, 0)
            if time.time() < cooldown_until:
                remaining = int(cooldown_until - time.time())
                return False, f"{symbol} 冷卻中({remaining}s)"

        return True, "OK"

    def can_open_position(self, current_positions: int) -> Tuple[bool, str]:
        if current_positions >= self.max_positions:
            return False, f"已達最大持倉 {self.max_positions}"
        return self.can_trade()

    # -------------------------------------------------------------------------
    # Record outcomes
    # -------------------------------------------------------------------------

    def record_trade(self, pnl: float, symbol: Optional[str] = None,
                     is_stop_loss: bool = False) -> None:
        """Record a completed trade and update state."""
        self.daily_pnl += pnl
        self.trade_count += 1
        self.current_balance += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Per-symbol cooldown after any exit
        if symbol:
            self._symbol_cooldown[symbol] = time.time() + self.re_entry_cooldown_sec
            # Stop-loss ban: same symbol, no re-entry today
            if is_stop_loss:
                self._symbol_stop_ban[symbol] = time.time() + 86400  # ban until next day

        # v2: 3 consecutive losses → day_stopped (not auto-unlock)
        if self.consecutive_losses >= self.consecutive_loss_limit and not self.day_stopped:
            self.day_stopped = True
            self.day_stop_reason = f"連虧{self.consecutive_losses}筆 → 今日停止交易"
            logger.warning(f"⛔ {self.day_stop_reason}")

        # Daily loss limit check
        if self.daily_pnl <= -self.max_daily_loss and not self.day_stopped:
            self.day_stopped = True
            self.day_stop_reason = f"日損上限 {self.daily_pnl:,.0f}/{-self.max_daily_loss:,.0f}"
            logger.warning(f"⛔ {self.day_stop_reason}")

    # -------------------------------------------------------------------------
    # Position sizing
    # -------------------------------------------------------------------------

    def calculate_position_size_usd(
        self,
        entry_price: float,
        stop_price: float,
        win_rate: float = 0.5,
        smc_score: int = 75,
    ) -> int:
        """
        Calculate position size in shares for USD stocks.
        Returns number of shares (integer).
        """
        if entry_price <= 0 or stop_price <= 0:
            return 0
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        # Kelly criterion
        R = abs(entry_price - stop_price) * 3.0 / abs(entry_price - stop_price)  # R ratio
        kelly_f = win_rate - (1 - win_rate) / R if R > 0 else 0
        kelly_f = max(0.0, min(kelly_f, 0.5))
        position_value = self.current_balance * kelly_f * self.kelly_fraction

        # Cap by single position risk
        max_risk_value = self.current_balance * self.max_single_risk_pct
        max_shares_by_risk = int(max_risk_value / risk_per_share) if risk_per_share > 0 else 0

        # Cap by max 10% of account
        max_shares_by_capital = int(self.current_balance * 0.10 / entry_price)

        # Volatility scaling
        atr_pct = risk_per_share / entry_price
        if atr_pct > 0.03:
            vol_scale = 0.5
        elif atr_pct > 0.015:
            vol_scale = 1.0
        else:
            vol_scale = 1.33

        shares = int((position_value / entry_price) * vol_scale)
        shares = min(shares, max_shares_by_risk, max_shares_by_capital)

        # Score-based adjustment
        if smc_score < 80:
            shares = min(shares, 10)
        elif smc_score >= 90:
            shares = min(shares, 50)
        else:
            shares = min(shares, 25)

        return max(1, shares) if shares > 0 else 0

    def calculate_position_size_tw(
        self,
        entry_price: float,
        stop_price: float,
        smc_score: int = 75,
    ) -> int:
        """
        Calculate position size in lots (1 lot = 1000 shares) for TW stocks.
        Returns number of shares (multiple of 1000).
        """
        if entry_price <= 0 or stop_price <= 0:
            return 0
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        # Max risk per trade
        max_risk = self.current_balance * self.max_single_risk_pct
        max_shares = int(max_risk / risk_per_share)
        max_lots = max_shares // 1000

        # Score-based limit
        if smc_score < 80:
            max_lots = min(max_lots, 1)  # weak signal: 1 lot
        elif smc_score >= 85:
            max_lots = min(max_lots, 2)  # strong signal: up to 2 lots
        else:
            max_lots = min(max_lots, 1)

        # Price guard: 1 lot × price <= 10% of balance
        price_limit_lots = int(self.current_balance * 0.10 / (entry_price * 1000))
        max_lots = min(max_lots, max(1, price_limit_lots))

        return max(1000, max_lots * 1000) if max_lots >= 1 else 0
