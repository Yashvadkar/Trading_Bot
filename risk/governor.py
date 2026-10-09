"""Deterministic Software 1.0 Risk Governor (Agency Reality Checker).

The ultimate gatekeeper between AI model signals and broker execution:
- Fractional Kelly Criterion Capital Allocation: f^* = kappa * (mu / sigma^2)
- Order-to-Trade Ratio (OTR) Regulator: Rolling 24h tracker with throttle and regulatory kill-switch
- Almgren-Chriss Optimal Execution: Parent order slicing into TWAP/POV blocks
- Dynamic EVT Tail-Risk Dampener: Tightens allocation during fat-tailed market stress
- Ratchet Trailing Stop-Loss: Locks in profits when unrealized gains exceed 3%
- Hard maximum 20% capital concentration in any single equity.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from config import RiskConfig


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionActionType(str, Enum):
    HOLD = "hold"
    EXIT = "exit"
    UPDATE_STOP = "update_stop"


@dataclass(frozen=True)
class OrderDirective:
    ticker: str
    side: OrderSide
    shares: int
    limit_price: float
    stop_loss_price: float
    dollar_allocation: float
    risk_pct_actual: float
    is_authorized: bool
    rejection_reason: str | None = None
    child_slices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class PositionDirective:
    ticker: str
    action_type: PositionActionType
    current_price: float
    new_stop_price: float
    shares: int
    unrealized_pnl_pct: float
    reason: str


class OTRTracker:
    """Regulatory Order-to-Trade Ratio (OTR) monitor."""

    def __init__(self, soft_limit: float = 35.0, kill_limit: float = 45.0, window_sec: int = 86400):
        self.soft_limit = soft_limit
        self.kill_limit = kill_limit
        self.window_sec = window_sec
        self.orders: list[float] = []
        self.trades: list[float] = []

    def record_order(self) -> None:
        self.orders.append(time.time())

    def record_trade(self) -> None:
        self.trades.append(time.time())

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        self.orders = [t for t in self.orders if t >= cutoff]
        self.trades = [t for t in self.trades if t >= cutoff]

    def get_current_otr(self) -> float:
        now = time.time()
        self._prune(now)
        order_count = len(self.orders)
        trade_count = len(self.trades)
        if trade_count == 0:
            return float(order_count)
        return float(order_count) / float(trade_count)

    def check_safety(self) -> tuple[bool, str | None]:
        now = time.time()
        self._prune(now)
        otr = self.get_current_otr()
        # Enforce check once at least 10 orders have been submitted
        if len(self.orders) >= 10:
            if otr >= self.kill_limit:
                return False, f"OTR Regulatory Kill-Switch engaged: ratio {otr:.1f}:1 exceeds exchange penalty ceiling ({self.kill_limit}:1)"
            if otr >= self.soft_limit:
                return True, f"OTR Warning: ratio {otr:.1f}:1 approaching threshold ({self.soft_limit}:1)"
        return True, None


class AlmgrenChrissSlicer:
    """Almgren-Chriss Optimal Execution Parent-Order Slicer (TWAP/POV)."""

    @staticmethod
    def slice_order(
        total_shares: int,
        price: float,
        adv: float = 10_000_000.0,
        max_pov_pct: float = 0.05,
        threshold_usd: float = 10_000.0
    ) -> list[int]:
        """Slices parent orders exceeding $10k to minimize Implementation Shortfall."""
        total_usd = total_shares * price
        if total_usd < threshold_usd or total_shares <= 10:
            return [total_shares]

        daily_shares = max(adv / max(price, 1.0), 1000.0)
        max_slice = max(10, int(daily_shares * max_pov_pct / 10.0))  # 10% of POV block
        if max_slice >= total_shares:
            return [total_shares]

        slices = []
        rem = total_shares
        while rem > 0:
            cur = min(rem, max_slice)
            slices.append(cur)
            rem -= cur
        return slices


def calculate_fractional_kelly_capital(
    returns: np.ndarray | list[float] | None,
    portfolio_equity: float,
    kappa: float = 0.15,
    max_leverage: float = 1.5,
    max_capital_pct: float = 0.20
) -> float:
    """Calculates capital allocation using Fractional Kelly Criterion.
    
    Formula: f^* = kappa * (mu / sigma^2)
    Tempered by parameter estimation uncertainty kappa (0.15) and gross leverage ceiling.
    """
    if returns is None:
        return portfolio_equity * 0.15

    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 5:
        return portfolio_equity * 0.15

    mu = float(np.mean(arr))
    sigma_sq = float(np.var(arr))

    # Reject if variance is near 0 or historical drift is non-positive
    if sigma_sq <= 1e-8 or np.isnan(sigma_sq) or mu <= 0.0:
        return 0.0

    full_kelly = mu / sigma_sq
    target_fraction = kappa * full_kelly
    constrained_fraction = float(np.clip(target_fraction, 0.0, min(max_leverage, max_capital_pct)))

    return float(portfolio_equity * constrained_fraction)


class RiskGovernor:
    """Mathematical gatekeeper authorizing, sizing, or rejecting trade orders."""

    def __init__(self, risk_cfg: RiskConfig):
        self.cfg = risk_cfg
        self.otr_tracker = OTRTracker(
            soft_limit=getattr(risk_cfg, "otr_soft_threshold", 35.0),
            kill_limit=getattr(risk_cfg, "otr_kill_threshold", 45.0),
            window_sec=getattr(risk_cfg, "otr_window_seconds", 86400)
        )
        self.slicer = AlmgrenChrissSlicer()

    def evaluate_order(
        self,
        ticker: str,
        action: int,                  # 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
        current_price: float,
        atr: float,
        portfolio_equity: float,
        available_cash: float,
        current_positions_count: int,
        is_already_held: bool,
        circuit_breaker_active: bool,
        evt_var_99: float = 0.0,
        returns_history: np.ndarray | None = None
    ) -> OrderDirective:
        """Evaluates trade intent and returns an authorized, sized order or rejection."""
        # 1. Action 0: HOLD
        if action == 0:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason="Action is HOLD"
            )

        # 2. Action 3: EXIT
        if action == 3:
            if not is_already_held:
                return OrderDirective(
                    ticker=ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=current_price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Exit requested but ticker is not in portfolio"
                )
            self.otr_tracker.record_order()
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.SELL,
                shares=0,  # 0 indicates close entire position
                limit_price=current_price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=True,
                rejection_reason=None
            )

        # 3. Actions 1 & 2: BUY
        if circuit_breaker_active:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason="Circuit breaker active (daily drawdown halt)"
            )

        # OTR Regulatory Gate Check
        otr_safe, otr_msg = self.otr_tracker.check_safety()
        if not otr_safe:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason=otr_msg or "OTR Kill-Switch engaged"
            )

        if is_already_held:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason=f"Position in {ticker} is already open"
            )

        if current_positions_count >= self.cfg.max_concurrent_positions:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason=f"Max concurrent positions reached ({self.cfg.max_concurrent_positions})"
            )

        # Action multiplier: 1 -> 0.5x, 2 -> 1.0x
        scale_factor = 0.5 if action == 1 else 1.0
        # Extreme Value Theory (EVT) tail-risk dampener: if tail loss > 5%, reduce size by 35%
        if evt_var_99 > 0.05:
            scale_factor *= 0.65

        risk_budget_pct = self.cfg.max_risk_per_trade * scale_factor
        dollar_risk = portfolio_equity * risk_budget_pct

        # Trailing stop distance = max(ATR * multiplier, EVT tail risk)
        base_stop_dist = max(atr * self.cfg.atr_multiplier_stop, current_price * 0.01)
        if evt_var_99 > 0.0:
            stop_dist = max(base_stop_dist, current_price * evt_var_99)
        else:
            stop_dist = base_stop_dist
        stop_loss_price = max(0.01, current_price - stop_dist)

        # Shares to risk exact dollar budget
        raw_shares = int(dollar_risk / stop_dist)
        if raw_shares <= 0:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=stop_loss_price,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason="Share sizing rounded to 0 under ATR risk budget"
            )

        # Fractional Kelly Capital Allocation
        kelly_fraction = getattr(self.cfg, "kelly_fraction", 0.15)
        max_lev = getattr(self.cfg, "max_leverage", 1.5)
        kelly_capital = calculate_fractional_kelly_capital(
            returns=returns_history,
            portfolio_equity=portfolio_equity,
            kappa=kelly_fraction,
            max_leverage=max_lev,
            max_capital_pct=0.20
        )

        # Cap single position to max Fractional Kelly allocation, available cash, or 20% equity
        if returns_history is not None and len(returns_history) >= 5:
            if kelly_capital <= 0.0:
                return OrderDirective(
                    ticker=ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=current_price,
                    stop_loss_price=stop_loss_price,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Fractional Kelly allocated 0 capital (negative historical drift/high variance)"
                )
            max_capital_for_pos = min(portfolio_equity * 0.20, available_cash * 0.95, kelly_capital)
        else:
            max_capital_for_pos = min(portfolio_equity * 0.20, available_cash * 0.95)

        max_shares_capital = int(max_capital_for_pos / current_price) if current_price > 0 else 0
        final_shares = min(raw_shares, max_shares_capital)

        if final_shares <= 0:
            return OrderDirective(
                ticker=ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=current_price,
                stop_loss_price=stop_loss_price,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason="Capital constraints capped shares to 0"
            )

        actual_allocation = float(final_shares * current_price)
        actual_risk_pct = float((final_shares * stop_dist) / portfolio_equity) if portfolio_equity > 0 else 0.0

        # Almgren-Chriss Order Slicing for parent orders
        child_slices = self.slicer.slice_order(final_shares, current_price, adv=self.cfg.min_avg_daily_volume_usd)

        # Record authorized order in OTR tracker
        self.otr_tracker.record_order()

        return OrderDirective(
            ticker=ticker,
            side=OrderSide.BUY,
            shares=final_shares,
            limit_price=current_price,
            stop_loss_price=round(stop_loss_price, 2),
            dollar_allocation=round(actual_allocation, 2),
            risk_pct_actual=round(actual_risk_pct, 4),
            is_authorized=True,
            rejection_reason=None,
            child_slices=child_slices
        )

    def evaluate_held_position(
        self,
        ticker: str,
        shares: int,
        entry_price: float,
        current_price: float,
        atr: float,
        unrealized_pnl_pct: float,
        debate_action: int,
        current_stop_price: float
    ) -> PositionDirective:
        """Monitors open positions, triggering early exits or ratcheting trailing stops."""
        # Action 3: EXIT signal from debate arena
        if debate_action == 3:
            self.otr_tracker.record_trade()
            return PositionDirective(
                ticker=ticker,
                action_type=PositionActionType.EXIT,
                current_price=current_price,
                new_stop_price=0.0,
                shares=shares,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason="Adversarial debate arena consensus generated early EXIT signal."
            )

        # Dynamic Ratchet Trailing Stop:
        # If profit > 3%, raise stop to (current_price - 1.5 * ATR)
        if unrealized_pnl_pct >= 0.03:
            proposed_stop = round(current_price - (1.5 * atr), 2)
            if proposed_stop > current_stop_price:
                return PositionDirective(
                    ticker=ticker,
                    action_type=PositionActionType.UPDATE_STOP,
                    current_price=current_price,
                    new_stop_price=proposed_stop,
                    shares=shares,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    reason=f"Profit lock ratcheted stop upward from ${current_stop_price:.2f} to ${proposed_stop:.2f} (+{unrealized_pnl_pct*100:.1f}% gain)."
                )

        return PositionDirective(
            ticker=ticker,
            action_type=PositionActionType.HOLD,
            current_price=current_price,
            new_stop_price=current_stop_price,
            shares=shares,
            unrealized_pnl_pct=unrealized_pnl_pct,
            reason="Position within normal risk bounds; holding."
        )
