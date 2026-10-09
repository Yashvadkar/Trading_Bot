"""Deterministic Software 1.0 Risk Governor (Agency Reality Checker).

The ultimate gatekeeper between AI model signals and broker execution:
- Position Sizing = (Portfolio Equity * Max Risk %) / (ATR * Multiplier)
- Maximum 20% capital concentration in any single equity.
- Enforces maximum concurrent open positions.
- Trailing stop-loss calculated at entry: Entry Price - (ATR * Multiplier).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
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


@dataclass(frozen=True)
class PositionDirective:
    ticker: str
    action_type: PositionActionType
    current_price: float
    new_stop_price: float
    shares: int
    unrealized_pnl_pct: float
    reason: str


class RiskGovernor:
    """Mathematical gatekeeper authorizing, sizing, or rejecting trade orders."""

    def __init__(self, risk_cfg: RiskConfig):
        self.cfg = risk_cfg

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
        circuit_breaker_active: bool
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
        risk_budget_pct = self.cfg.max_risk_per_trade * scale_factor
        dollar_risk = portfolio_equity * risk_budget_pct

        # Trailing stop distance = ATR * multiplier
        stop_dist = max(atr * self.cfg.atr_multiplier_stop, current_price * 0.01)
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

        # Cap single position to max 20% of portfolio equity or available cash
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
                rejection_reason="Insufficient cash for minimum 1 share allocation"
            )

        allocation_usd = final_shares * current_price
        actual_risk_usd = final_shares * stop_dist
        actual_risk_pct = actual_risk_usd / portfolio_equity if portfolio_equity > 0 else 0.0

        return OrderDirective(
            ticker=ticker,
            side=OrderSide.BUY,
            shares=final_shares,
            limit_price=current_price,
            stop_loss_price=round(stop_loss_price, 2),
            dollar_allocation=round(allocation_usd, 2),
            risk_pct_actual=round(actual_risk_pct, 4),
            is_authorized=True,
            rejection_reason=None
        )

    def evaluate_held_position(
        self,
        ticker: str,
        shares: int,
        entry_price: float,
        current_price: float,
        atr: float,
        unrealized_pnl_pct: float,
        debate_action: int,        # 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
        current_stop_price: float
    ) -> PositionDirective:
        """Evaluates an existing held position for neural early-exit or ratchet trailing stops."""
        # 1. Neural Exit: If Bear network or consensus decides EXIT (3), close immediately
        if debate_action == 3:
            return PositionDirective(
                ticker=ticker,
                action_type=PositionActionType.EXIT,
                current_price=current_price,
                new_stop_price=0.0,
                shares=shares,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason="Neural debate triggered EXIT signal (bearish trend/distribution)"
            )

        # 2. Ratchet Trailing Stop:
        # If position is in profit (e.g. >= +2.5% gain), calculate a ratcheted trailing stop
        if unrealized_pnl_pct >= 0.025:
            trail_distance = max(atr * 1.5, current_price * 0.015)
            proposed_stop = round(current_price - trail_distance, 2)

            # Ensure the stop is only ratcheted UPWARD (protecting capital/locking profit)
            if proposed_stop > current_stop_price + 0.10:
                status_msg = "Profit lock ratcheted" if proposed_stop > entry_price else "Trailing stop raised towards break-even"
                return PositionDirective(
                    ticker=ticker,
                    action_type=PositionActionType.UPDATE_STOP,
                    current_price=current_price,
                    new_stop_price=proposed_stop,
                    shares=shares,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    reason=f"{status_msg} (+{unrealized_pnl_pct*100:.1f}% gain, stop raised to ${proposed_stop:.2f})"
                )

        return PositionDirective(
            ticker=ticker,
            action_type=PositionActionType.HOLD,
            current_price=current_price,
            new_stop_price=current_stop_price,
            shares=shares,
            unrealized_pnl_pct=unrealized_pnl_pct,
            reason="Position healthy; holding"
        )
