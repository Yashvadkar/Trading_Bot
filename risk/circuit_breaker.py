"""Portfolio Circuit Breaker & Emergency Kill-Switch (Agency Reality Checker).

Monitors peak equity and daily drawdown to prevent runaway portfolio loss.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class BreakerStatus(str, Enum):
    NORMAL = "NORMAL"
    HALTED = "HALTED"                    # Daily loss > 3%: No new buys allowed
    EMERGENCY = "EMERGENCY_LIQUIDATION"  # Daily loss > 5%: Exit all open positions


@dataclass
class CircuitBreakerState:
    status: BreakerStatus
    initial_equity: float
    current_equity: float
    daily_drawdown_pct: float
    message: str


class CircuitBreaker:
    """Deterministic circuit breaker safeguarding total capital."""

    def __init__(self, halt_drawdown_limit: float = 0.03, emergency_limit: float = 0.05):
        self.halt_limit = halt_drawdown_limit
        self.emergency_limit = emergency_limit
        self.day_start_equity: float | None = None
        self.peak_equity: float = 0.0

    def update_equity(self, current_equity: float) -> CircuitBreakerState:
        """Update equity and evaluate circuit breaker threshold."""
        if current_equity <= 0.0:
            return CircuitBreakerState(
                status=BreakerStatus.HALTED,
                initial_equity=0.0,
                current_equity=0.0,
                daily_drawdown_pct=1.0,
                message="Invalid or zero equity reported."
            )

        if self.day_start_equity is None:
            self.day_start_equity = current_equity
            self.peak_equity = current_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Daily drawdown measured from beginning-of-day equity
        daily_loss = max(0.0, self.day_start_equity - current_equity)
        drawdown_pct = daily_loss / self.day_start_equity if self.day_start_equity > 0 else 0.0

        if drawdown_pct >= self.emergency_limit:
            status = BreakerStatus.EMERGENCY
            msg = f"EMERGENCY: Daily drawdown {drawdown_pct*100:.2f}% breached {self.emergency_limit*100:.1f}% limit! Halting and liquidating."
        elif drawdown_pct >= self.halt_limit:
            status = BreakerStatus.HALTED
            msg = f"WARNING: Daily drawdown {drawdown_pct*100:.2f}% breached {self.halt_limit*100:.1f}% halt threshold. Buys paused."
        else:
            status = BreakerStatus.NORMAL
            msg = f"Normal operation. Drawdown: {drawdown_pct*100:.2f}% (Limit: {self.halt_limit*100:.1f}%)"

        return CircuitBreakerState(
            status=status,
            initial_equity=self.day_start_equity,
            current_equity=current_equity,
            daily_drawdown_pct=drawdown_pct,
            message=msg
        )

    def can_open_new_position(self, current_equity: float) -> bool:
        """Returns True only if circuit breaker is in NORMAL state."""
        state = self.update_equity(current_equity)
        return state.status == BreakerStatus.NORMAL
