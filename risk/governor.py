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
from collections import deque
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
    shares: int | float
    limit_price: float
    stop_loss_price: float
    dollar_allocation: float
    risk_pct_actual: float
    is_authorized: bool
    rejection_reason: str | None = None
    child_slices: list[int | float] = field(default_factory=list)


@dataclass(frozen=True)
class PositionDirective:
    ticker: str
    action_type: PositionActionType
    current_price: float
    new_stop_price: float
    shares: int
    unrealized_pnl_pct: float
    reason: str


class SliceRoutingType(str, Enum):
    PASSIVE_LIMIT = "passive_limit"
    AGGRESSIVE_MARKET = "aggressive_market"


@dataclass(frozen=True)
class ChildSliceDirective:
    shares: int | float
    routing_type: SliceRoutingType
    fill_prob: float
    urgency_kappa: float


class OTRTracker:
    """Regulatory Order-to-Trade Ratio (OTR) monitor with vectorized SEBI/NSE proximity exemption.
    
    Orders placed within +-delta (default 0.75% for cash equities) of the Last Traded Price (LTP)
    are exempt from the high OTR penalty numerator to avoid penalizing near-market liquidity provision.
    Uses memory-contiguous numpy structured arrays for vectorized evaluation and deques for zero-copy pruning.
    """

    ORDER_DTYPE = np.dtype([
        ('timestamp', np.float64),
        ('price', np.float64),
        ('ltp', np.float64),
        ('exempt', np.bool_)
    ])

    def __init__(
        self,
        soft_limit: float = 35.0,
        kill_limit: float = 45.0,
        window_sec: int = 86400,
        proximity_delta: float = 0.0075
    ):
        self.soft_limit = soft_limit
        self.kill_limit = kill_limit
        self.window_sec = window_sec
        self.proximity_delta = proximity_delta
        self.non_exempt_orders: deque[float] = deque()
        self.exempt_orders: deque[float] = deque()
        self.trades: deque[float] = deque()
        self._order_records: list[tuple[float, float, float, bool]] = []

    def record_order(self, price: float | None = None, ltp: float | None = None) -> None:
        """Records an order timestamp, checking for proximity exemption."""
        now = time.time()
        p = float(price) if price is not None else 0.0
        l = float(ltp) if ltp is not None else 0.0
        is_exempt = False
        if price is not None and ltp is not None and l > 0:
            diff_pct = abs(p - l) / l
            if diff_pct <= self.proximity_delta:
                is_exempt = True

        self._order_records.append((now, p, l, is_exempt))
        if is_exempt:
            self.exempt_orders.append(now)
        else:
            self.non_exempt_orders.append(now)

    def record_batch_orders(
        self,
        prices: np.ndarray | list[float],
        ltps: np.ndarray | list[float],
        timestamps: np.ndarray | list[float] | None = None
    ) -> int:
        """Vectorized batch order registration using contiguous numpy arrays."""
        p_arr = np.asarray(prices, dtype=np.float64)
        l_arr = np.asarray(ltps, dtype=np.float64)
        n = len(p_arr)
        if n == 0:
            return 0
        now = time.time()
        t_arr = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else np.full(n, now, dtype=np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):
            valid_ltp = l_arr > 0
            diff_pct = np.zeros(n, dtype=np.float64)
            np.divide(np.abs(p_arr - l_arr), l_arr, out=diff_pct, where=valid_ltp)
            exempt = valid_ltp & (diff_pct <= self.proximity_delta)

        for i in range(n):
            t_val = float(t_arr[i])
            is_ex = bool(exempt[i])
            self._order_records.append((t_val, float(p_arr[i]), float(l_arr[i]), is_ex))
            if is_ex:
                self.exempt_orders.append(t_val)
            else:
                self.non_exempt_orders.append(t_val)
        return int(np.sum(exempt))

    def record_trade(self) -> None:
        self.trades.append(time.time())

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self.non_exempt_orders and self.non_exempt_orders[0] < cutoff:
            self.non_exempt_orders.popleft()
        while self.exempt_orders and self.exempt_orders[0] < cutoff:
            self.exempt_orders.popleft()
        while self.trades and self.trades[0] < cutoff:
            self.trades.popleft()
        while self._order_records and self._order_records[0][0] < cutoff:
            self._order_records.pop(0)

    def compute_vectorized_otr(self) -> float:
        """Vector-calculates OTR across rolling contiguous numpy structured array."""
        now = time.time()
        self._prune(now)
        if not self._order_records:
            return 0.0
        arr = np.array(self._order_records, dtype=self.ORDER_DTYPE)
        non_exempt_count = int(np.sum(~arr['exempt']))
        trade_count = len(self.trades)
        return float(non_exempt_count) / float(max(1, trade_count))

    def get_current_otr(self) -> float:
        now = time.time()
        self._prune(now)
        penalty_orders = len(self.non_exempt_orders)
        trade_count = len(self.trades)
        if trade_count == 0:
            return float(penalty_orders)
        return float(penalty_orders) / float(trade_count)

    def check_safety(self) -> tuple[bool, str | None]:
        now = time.time()
        self._prune(now)
        otr = self.get_current_otr()
        # Enforce check once at least 10 non-exempt orders have been submitted
        if len(self.non_exempt_orders) >= 10:
            if otr >= self.kill_limit:
                return False, f"OTR Regulatory Kill-Switch engaged: ratio {otr:.1f}:1 exceeds exchange penalty ceiling ({self.kill_limit}:1)"
            if otr >= self.soft_limit:
                return True, f"OTR Warning: ratio {otr:.1f}:1 approaching threshold ({self.soft_limit}:1)"
        return True, None


class AlmgrenChrissSlicer:
    """Almgren-Chriss Optimal Execution Parent-Order Slicer (TWAP/POV) with CVD & Cont-Larrard queue decay."""

    @staticmethod
    def calculate_urgency(
        volatility: float = 0.02,
        risk_aversion: float = 1e-6,
        temp_impact: float = 2.5e-7,
        cvd_divergence: float = 0.0,
        side: OrderSide = OrderSide.BUY
    ) -> float:
        """Calculates Almgren-Chriss urgency parameter kappa, modulated by CVD absorption.
        
        kappa = sqrt(lambda * sigma^2 / eta)
        - If selling into bullish absorption (passive buyer wall, cvd_divergence = 1.0),
          dampens urgency by 50% to avoid crossing the spread into passive liquidity.
        - If selling during aggressive seller exhaustion or aligned trend, boosts urgency by 30%.
        """
        base_urgency = float(np.sqrt((max(risk_aversion, 1e-8) * (volatility ** 2)) / max(temp_impact, 1e-8)))

        # Heuristic modulation via CVD Absorption Divergence
        if side == OrderSide.SELL and cvd_divergence > 0.0:
            # Passive buyers absorbing selling pressure at support -> throttle urgency
            multiplier = 0.50
        elif side == OrderSide.BUY and cvd_divergence < 0.0:
            # Passive sellers absorbing buying pressure at resistance -> throttle urgency
            multiplier = 0.50
        elif abs(cvd_divergence) > 0.0:
            # Momentum aligned with order -> front-load urgency
            multiplier = 1.30
        else:
            multiplier = 1.0

        return float(base_urgency * multiplier)

    @staticmethod
    def calculate_cont_larrard_fill_prob(queue_position: float, gamma: float = 0.5) -> float:
        """Cont-Larrard queue fill probability decay: P_fill = exp(-gamma * q_pos)."""
        q_pos = max(0.0, float(queue_position))
        return float(np.exp(-gamma * q_pos))

    @staticmethod
    def recalculate_urgency_on_spread_crossing(current_urgency: float, spread_penalty: float = 0.25) -> float:
        """Dynamically throttles execution urgency for remaining inventory after crossing the spread.
        
        eta_adj = eta * (1 + spread_penalty)
        kappa_adj = kappa / sqrt(1 + spread_penalty)
        Throttles subsequent slices to avoid compounding market impact and adverse selection.
        """
        penalty = max(0.0, float(spread_penalty))
        return float(current_urgency / np.sqrt(1.0 + penalty))

    @classmethod
    def slice_order(
        cls,
        total_shares: int | float,
        price: float,
        adv: float = 10_000_000.0,
        max_pov_pct: float = 0.05,
        threshold_usd: float = 10_000.0,
        volatility: float = 0.02,
        cvd_divergence: float = 0.0,
        side: OrderSide = OrderSide.BUY
    ) -> list[int | float]:
        """Slices parent orders into discrete child orders with CVD-modulated urgency."""
        total_usd = float(total_shares) * float(price)
        if total_usd < threshold_usd or total_shares <= 10:
            return [total_shares]

        # Step 3: Almgren-Chriss Retail Execution Bypass
        # For retail-sized orders (< 1% of 5-minute ADV), bypass slicing to avoid spread crossing drag
        # 78 five-minute periods in regular trading hours (6.5 hours)
        adv_5m_shares = max(100.0, (float(adv) / max(float(price), 1.0)) / 78.0)
        if float(total_shares) < 0.01 * adv_5m_shares:
            return [total_shares]

        daily_shares = max(adv / max(price, 1.0), 1000.0)
        base_slice = max(10, int(daily_shares * max_pov_pct / 10.0))  # 10% of POV block

        # Modulate slice size by Almgren-Chriss urgency
        urgency = cls.calculate_urgency(
            volatility=volatility,
            cvd_divergence=cvd_divergence,
            side=side
        )
        # When urgency is high, front-load child slice sizes; when low, smaller TWAP slices
        urgency_multiplier = float(np.clip(1.0 + (urgency * 5.0 - 0.2), 0.5, 2.0))
        max_slice = max(10, int(base_slice * urgency_multiplier))

        if max_slice >= total_shares:
            return [total_shares]

        slices = []
        rem = total_shares
        while rem > 0:
            cur = min(rem, max_slice)
            slices.append(cur)
            rem -= cur
        return slices

    @classmethod
    def route_child_slices(
        cls,
        total_shares: int | float,
        price: float,
        adv: float = 10_000_000.0,
        max_pov_pct: float = 0.05,
        threshold_usd: float = 10_000.0,
        volatility: float = 0.02,
        cvd_divergence: float = 0.0,
        side: OrderSide = OrderSide.BUY,
        queue_position: float = 0.5,
        gamma: float = 0.5
    ) -> list[ChildSliceDirective]:
        """Generates child slice directives with dynamic passive limit vs aggressive market routing."""
        slices = cls.slice_order(
            total_shares=total_shares,
            price=price,
            adv=adv,
            max_pov_pct=max_pov_pct,
            threshold_usd=threshold_usd,
            volatility=volatility,
            cvd_divergence=cvd_divergence,
            side=side
        )
        urgency = cls.calculate_urgency(
            volatility=volatility,
            cvd_divergence=cvd_divergence,
            side=side
        )
        p_fill = cls.calculate_cont_larrard_fill_prob(queue_position, gamma=gamma)

        # Dynamic routing switch: if fill probability < 0.40 (deep queue with adverse drift risk)
        # or aggressive momentum alignment with elevated urgency, switch to AGGRESSIVE_MARKET
        if p_fill < 0.40 or (side == OrderSide.BUY and cvd_divergence > 0.5 and urgency > 0.15):
            routing = SliceRoutingType.AGGRESSIVE_MARKET
        else:
            routing = SliceRoutingType.PASSIVE_LIMIT

        directives = []
        current_urgency = urgency
        for s in slices:
            directives.append(
                ChildSliceDirective(
                    shares=s,
                    routing_type=routing,
                    fill_prob=p_fill,
                    urgency_kappa=current_urgency
                )
            )
            if routing == SliceRoutingType.AGGRESSIVE_MARKET:
                current_urgency = cls.recalculate_urgency_on_spread_crossing(current_urgency)

        return directives


def calculate_conformal_volatility(
    residuals: np.ndarray | list[float] | None,
    alpha: float = 0.20,
    z_score: float = 1.2816
) -> float:
    """Extracts scale sigma from the empirical (1 - alpha) rolling conformal prediction interval.
    
    Formula: sigma_hat = q_eff / z_(1 - alpha/2)
    where q_eff is the (1 - alpha) empirical quantile of absolute forecast residuals.
    """
    if residuals is None:
        return 0.02
    arr = np.asarray(residuals, dtype=np.float64)
    if len(arr) < 5:
        return 0.02
    abs_res = np.abs(arr)
    # Target coverage: 1 - alpha (e.g. 0.80 for alpha=0.20)
    q_eff = float(np.quantile(abs_res, 1.0 - alpha))
    sigma_hat = float(max(q_eff / max(z_score, 1e-4), 1e-4))
    return sigma_hat


def calculate_fractional_kelly_capital(
    returns: np.ndarray | list[float] | None,
    portfolio_equity: float,
    kappa: float = 0.15,
    max_leverage: float = 1.5,
    max_capital_pct: float = 0.20,
    conformal_residuals: np.ndarray | list[float] | None = None,
    conformal_alpha: float = 0.20,
    conformal_z_score: float = 1.2816
) -> float:
    """Calculates capital allocation using Fractional Kelly with Conformal Uncertainty.
    
    Formula: f^* = kappa * (mu / sigma_eff^2)
    When conformal residuals are available, sigma_eff is derived directly from the
    conformal prediction interval width rather than naive historical sample variance.
    """
    if returns is None:
        return portfolio_equity * 0.15

    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 5:
        return portfolio_equity * 0.15

    mu = float(np.mean(arr))

    if conformal_residuals is not None and len(conformal_residuals) >= 5:
        sigma_conf = calculate_conformal_volatility(conformal_residuals, alpha=conformal_alpha, z_score=conformal_z_score)
        sigma_sq = float(sigma_conf ** 2)
    else:
        sigma_sq = float(np.var(arr))

    # Reject if variance is near 0 or historical drift is non-positive
    if sigma_sq <= 1e-8 or np.isnan(sigma_sq) or mu <= 0.0:
        return 0.0

    full_kelly = mu / sigma_sq
    target_fraction = kappa * full_kelly
    constrained_fraction = float(np.clip(target_fraction, 0.0, min(max_leverage, max_capital_pct)))

    return float(portfolio_equity * constrained_fraction)


def allocate_portfolio_with_cash_preservation(
    target_fractions: dict[str, float],
    max_leverage: float = 1.5,
    uncertainty_scale: float = 1.0,
    conformal_penalties: dict[str, float] | None = None,
    include_cash: bool = False
) -> dict[str, float]:
    """Allocates portfolio weights preserving CASH during systemic or asset-level uncertainty.
    
    Prevents the uniform gross leverage washout trap:
    1. Computes unconstrained targets f_i^*.
    2. Dampens by conformal model uncertainty phi_i = max(1.0, sigma_conf / sigma_baseline).
    3. Moves the unallocated capital directly into a dedicated CASH allocation bucket:
       w_cash = max(0.0, 1.0 - sum(f_i^* / phi_i)).
    4. Applies gross leverage caps strictly to invested allocations without eroding the cash buffer.
    """
    if not target_fractions:
        return {"CASH": 1.0} if include_cash else {}

    effective_gross_cap = max_leverage / max(1.0, uncertainty_scale)

    if conformal_penalties:
        penalized_targets = {}
        for sym, f_star in target_fractions.items():
            phi_i = max(1.0, conformal_penalties.get(sym, uncertainty_scale))
            penalized_targets[sym] = max(0.0, f_star) / phi_i
        total_requested = sum(penalized_targets.values())
        if total_requested <= 0.0:
            result = {k: 0.0 for k in target_fractions}
        elif total_requested <= effective_gross_cap:
            result = penalized_targets
        else:
            scale = effective_gross_cap / total_requested
            result = {k: w * scale for k, w in penalized_targets.items()}
    else:
        total_requested = sum(max(0.0, f) for f in target_fractions.values())
        if total_requested <= 0.0:
            result = {k: 0.0 for k in target_fractions}
        elif total_requested <= effective_gross_cap:
            result = {k: max(0.0, f) for k, f in target_fractions.items()}
        else:
            scale = effective_gross_cap / total_requested
            result = {k: max(0.0, f) * scale for k, f in target_fractions.items()}

    if include_cash:
        total_invested = sum(result.values())
        w_cash = max(0.0, 1.0 - total_invested)
        result_with_cash = dict(result)
        result_with_cash["CASH"] = w_cash
        return result_with_cash

    return result


@dataclass
class TradeOutcome:
    pnl: float
    r_multiple: float
    is_win: bool
    timestamp: float = field(default_factory=time.time)


class ProgressiveExposureTracker:
    """Goverdhan 'Gon' Gajjala / Mark Minervini Progressive Exposure Risk Governor.
    
    Dynamically modulates the risk budget ceiling R_eff = R_base * M_prog across 3 regimes:
    1. Contraction (Defensive): >= 2 consecutive losses or win rate < 25% -> M_prog = 0.50
    2. Neutral: Baseline conditions -> M_prog = 1.00
    3. Expansion (Offensive): >= 2 consecutive wins with avg win >= 2.0R AND market uptrend -> M_prog = 1.50
    
    Hard invariant: Single-position Kelly fraction kappa is strictly capped at kappa_max = 0.25 (Quarter-Kelly)
    to prevent hyper-Kelly geometric decay (f > f*).
    """

    def __init__(
        self,
        base_risk_pct: float = 0.005,  # 0.5% base risk per trade
        max_multiplier: float = 1.50,   # Expansion ceiling: 0.75%
        min_multiplier: float = 0.50,   # Contraction floor: 0.25%
        window_size: int = 10,
        kappa_max: float = 0.25
    ):
        self.base_risk_pct = base_risk_pct
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.window_size = window_size
        self.kappa_max = kappa_max
        self.trade_history: deque[TradeOutcome] = deque(maxlen=window_size)
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def record_trade_outcome(self, pnl: float, r_multiple: float) -> None:
        """Records a closed trade's PnL and R-multiple."""
        is_win = (float(pnl) > 0.0 or float(r_multiple) > 0.0)
        outcome = TradeOutcome(pnl=float(pnl), r_multiple=float(r_multiple), is_win=is_win)
        self.trade_history.append(outcome)
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def compute_risk_multiplier(self, market_uptrend: bool = True) -> float:
        """Computes M_prog based on trailing performance and market index filter."""
        if len(self.trade_history) == 0:
            return 1.0

        # Contraction check: >= 2 consecutive losses
        if self.consecutive_losses >= 2:
            return self.min_multiplier

        # Rolling win rate check across rolling window (if >= 4 trades)
        if len(self.trade_history) >= 4:
            wins = sum(1 for t in self.trade_history if t.is_win)
            win_rate = float(wins) / float(len(self.trade_history))
            if win_rate < 0.25:
                return self.min_multiplier

        # Expansion check: >= 2 consecutive wins, avg win >= 2.0R, and confirmed market uptrend (SPY > 21 EMA)
        if self.consecutive_wins >= 2 and market_uptrend:
            recent_wins = [t.r_multiple for t in self.trade_history if t.is_win]
            if recent_wins and (sum(recent_wins) / float(len(recent_wins))) >= 2.0:
                return self.max_multiplier

        return 1.0

    def get_effective_risk_pct(self, market_uptrend: bool = True) -> float:
        """Returns R_eff = R_base * M_prog."""
        multiplier = self.compute_risk_multiplier(market_uptrend=market_uptrend)
        return float(self.base_risk_pct * multiplier)

    def get_effective_kappa(self, requested_kappa: float = 0.15) -> float:
        """Strictly caps Kelly fraction at Quarter-Kelly (kappa <= 0.25) to avoid geometric ruin."""
        return float(min(requested_kappa, self.kappa_max))


@dataclass
class CandidateIntent:
    ticker: str
    price: float
    atr: float
    stop_dist: float
    action: int = 2
    debate_confidence: float = 0.5
    sortino_ratio: float = 1.0
    returns: np.ndarray | list[float] | None = None
    cvd_divergence: float = 0.0
    evt_var_99: float = 0.0
    conformal_residuals: np.ndarray | None = None


@dataclass(frozen=True)
class BatchAllocationResult:
    authorized_directives: list[OrderDirective]
    rejected_directives: list[OrderDirective]
    total_allocated_capital: float
    cash_pool: float
    weights: dict[str, float]
    selected_tickers: list[str]


class PortfolioCapitalAllocator:
    """Multi-Asset Covariance Kelly & Small-Account Cardinality-Constrained Allocator.
    
    Implements:
    1. Small-account cardinality constraints (K <= 2 for < $5k, K <= 3 for < $10k).
    2. Conviction ranking based on debate consensus confidence and Sortino ratio.
    3. Multi-Asset Covariance Kelly Vector allocation solving F^* = kappa * Sigma_LW^-1 * M.
    4. Ledoit-Wolf RMT covariance shrinkage ensuring strictly well-conditioned positive definiteness.
    5. Global Cash Pool ceiling: sum(Capital_i) <= Available Cash * 0.95 (Zero cash overdraw).
    6. Dual Constraint Position Sizing: min(Capital_i / Price_i, Equity * R_eff / StopDist_i).
    7. Fractional share precision down to 0.001 shares for small accounts.
    """

    def __init__(self, governor: RiskGovernor):
        self.governor = governor

    @staticmethod
    def solve_constrained_kelly(
        cov: np.ndarray,
        kappa: float,
        m_vec: np.ndarray,
        min_weight: float = 0.02,
        max_weight: float = 0.25,
        max_total: float = 1.0
    ) -> np.ndarray:
        """Solves Box-and-Budget Constrained Quadratic Program for Multi-Asset Kelly:
        
        min_f  (1/2) f^T cov f - kappa * m^T f
        s.t.   min_weight <= f_i <= max_weight
               sum(f_i) <= max_total
               
        Uses scipy.optimize.minimize (SLSQP) with exact analytical gradient.
        Falls back to pure NumPy Projected Gradient Descent (PGD) with dual bisection.
        """
        n = len(m_vec)
        if n == 0:
            return np.array([], dtype=np.float64)
        if n == 1:
            return np.array([float(np.clip(kappa * m_vec[0] / max(cov[0, 0], 1e-6), min_weight, max_weight))], dtype=np.float64)

        # 1. Primary: Scipy SLSQP with analytical gradient
        try:
            from scipy.optimize import minimize
            def loss(f):
                return 0.5 * float(f @ cov @ f) - float(kappa * np.dot(m_vec, f))
            def jac(f):
                return cov @ f - kappa * m_vec

            bounds = [(min(min_weight, max_weight), max_weight) for _ in range(n)]
            cons = [{'type': 'ineq', 'fun': lambda f: max_total - np.sum(f)}]
            f0 = np.full(n, min(max_weight, max_total / n), dtype=np.float64)
            res = minimize(loss, f0, jac=jac, bounds=bounds, constraints=cons, method='SLSQP')
            if res.success and np.all(np.isfinite(res.x)):
                return np.asarray(res.x, dtype=np.float64)
        except Exception:
            pass

        # 2. Resilient Pure NumPy Projected Gradient Descent (PGD) fallback
        f = np.full(n, min(max_weight, max_total / n), dtype=np.float64)
        L = float(np.linalg.norm(cov, ord=2))
        step_size = 1.0 / max(L, 1e-4)

        for _ in range(50):
            grad = cov @ f - kappa * m_vec
            f_next = f - step_size * grad
            f_next = np.clip(f_next, min_weight, max_weight)
            if np.sum(f_next) > max_total:
                low, high = 0.0, float(np.max(f_next))
                for _ in range(20):
                    mid = (low + high) / 2.0
                    f_proj = np.clip(f_next - mid, min_weight, max_weight)
                    if np.sum(f_proj) > max_total:
                        low = mid
                    else:
                        high = mid
                f_next = np.clip(f_next - high, min_weight, max_weight)
            if np.linalg.norm(f_next - f) < 1e-5:
                break
            f = f_next

        return np.asarray(f, dtype=np.float64)

    def allocate_batch_directives(
        self,
        candidates: list[CandidateIntent],
        portfolio_equity: float,
        available_cash: float,
        current_positions_count: int = 0,
        held_tickers: set[str] | list[str] | None = None,
        circuit_breaker_active: bool = False,
        market_uptrend: bool = True,
        allow_fractional: bool = True,
        settled_cash: float | None = None
    ) -> BatchAllocationResult:
        """Batch evaluate candidate intents with Multi-Asset Kelly & cash conservation."""
        held_set = set(held_tickers) if held_tickers else set()
        auth_directives: list[OrderDirective] = []
        rej_directives: list[OrderDirective] = []
        weights_dict: dict[str, float] = {}

        if not candidates:
            return BatchAllocationResult([], [], 0.0, max(0.0, available_cash * 0.95), {}, [])

        # 1. Systemic Circuit Breaker & Safety Checks
        if circuit_breaker_active:
            for c in candidates:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Circuit breaker active (daily drawdown halt)"
                ))
            return BatchAllocationResult([], rej_directives, 0.0, max(0.0, available_cash * 0.95), {}, [])

        otr_safe, otr_msg = self.governor.otr_tracker.check_safety()
        if not otr_safe:
            for c in candidates:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason=otr_msg or "OTR Kill-Switch engaged"
                ))
            return BatchAllocationResult([], rej_directives, 0.0, max(0.0, available_cash * 0.95), {}, [])

        if available_cash <= 0.0:
            for c in candidates:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Available cash is zero or negative"
                ))
            return BatchAllocationResult([], rej_directives, 0.0, 0.0, {}, [])

        # 2. Determine Small-Account Maximum Cardinality K
        cfg_max = getattr(self.governor.cfg, "max_concurrent_positions", 5)
        if portfolio_equity < 5000.0:
            max_k = min(2, cfg_max)
        elif portfolio_equity < 10000.0:
            max_k = min(3, cfg_max)
        else:
            max_k = cfg_max

        slots_available = max(0, max_k - current_positions_count)
        if slots_available == 0:
            for c in candidates:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason=f"Max concurrent positions reached ({max_k}) for account equity tier"
                ))
            return BatchAllocationResult([], rej_directives, 0.0, max(0.0, available_cash * 0.95), {}, [])

        # 3. Categorize candidates (EXIT, HOLD, ALREADY HELD, BUY)
        eligible_buys: list[CandidateIntent] = []
        for c in candidates:
            if c.ticker in held_set:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason=f"Position in {c.ticker} is already open"
                ))
            elif c.action == 0:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Action is HOLD"
                ))
            elif c.action == 3:
                # Early EXIT directive
                self.governor.otr_tracker.record_order(price=c.price, ltp=c.price)
                auth_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.SELL,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=True,
                    rejection_reason=None
                ))
            elif c.action in (1, 2) and c.price > 0:
                eligible_buys.append(c)
            else:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Invalid price or action"
                ))

        if not eligible_buys:
            return BatchAllocationResult(auth_directives, rej_directives, 0.0, max(0.0, available_cash * 0.95), {}, [])

        # 4. Rank Candidates by Conviction Score S_i = debate_confidence * Sortino
        def get_conviction_score(item: CandidateIntent) -> float:
            act_mult = 0.75 if item.action == 1 else 1.0
            return float(item.debate_confidence * max(0.1, item.sortino_ratio) * act_mult)

        eligible_buys.sort(key=get_conviction_score, reverse=True)

        selected = eligible_buys[:slots_available]
        dropped = eligible_buys[slots_available:]
        for d in dropped:
            rej_directives.append(OrderDirective(
                ticker=d.ticker,
                side=OrderSide.HOLD,
                shares=0,
                limit_price=d.price,
                stop_loss_price=0.0,
                dollar_allocation=0.0,
                risk_pct_actual=0.0,
                is_authorized=False,
                rejection_reason=f"Dropped by portfolio cardinality ranking (lower conviction score than top {slots_available})"
            ))

        # 5. Multi-Asset Covariance Kelly Vector Allocation
        m_assets = len(selected)
        cfg_kappa = getattr(self.governor.cfg, "kelly_fraction", 0.15)
        kappa = self.governor.progressive_tracker.get_effective_kappa(cfg_kappa)

        m_vec = []
        for c in selected:
            if c.returns is not None and len(c.returns) >= 5:
                m_vec.append(max(0.001, float(np.mean(c.returns))))
            else:
                m_vec.append(0.01 * max(0.5, float(c.debate_confidence)))
        m_vec = np.array(m_vec, dtype=np.float64)

        # Build Covariance Matrix (RMT Ledoit-Wolf or Regularized Diagonal)
        can_compute_rmt = False
        try:
            from features.quant_theories import QuantTheoryEngine
        except ImportError:
            QuantTheoryEngine = None

        if QuantTheoryEngine is not None and m_assets >= 2:
            all_returns = [c.returns for c in selected if c.returns is not None]
            if len(all_returns) == m_assets:
                min_len = min(len(r) for r in all_returns)
                if min_len >= 10:
                    try:
                        R = np.column_stack([np.asarray(c.returns[:min_len], dtype=np.float64) for c in selected])
                        shrunk_corr, _, _ = QuantTheoryEngine.clean_correlation_rmt(R, dynamic_shrinkage=True)
                        stds = np.std(R, axis=0, ddof=0)
                        stds[stds < 1e-4] = 1e-4
                        D = np.diag(stds)
                        cov_matrix = D @ shrunk_corr @ D
                        can_compute_rmt = True
                    except Exception:
                        can_compute_rmt = False

        if not can_compute_rmt:
            stds = []
            for c in selected:
                if c.returns is not None and len(c.returns) >= 5:
                    stds.append(max(1e-4, float(np.std(c.returns))))
                else:
                    stds.append(max(1e-4, float(c.atr / max(c.price, 1.0))))
            stds = np.array(stds, dtype=np.float64)
            cov_matrix = np.diag(stds ** 2)

        reg_cov = cov_matrix + 1e-6 * np.eye(m_assets)
        f_star = self.solve_constrained_kelly(
            cov=reg_cov,
            kappa=kappa,
            m_vec=m_vec,
            min_weight=0.02,
            max_weight=0.25,
            max_total=1.0
        )

        if np.sum(f_star) <= 1e-6:
            scores = np.array([get_conviction_score(c) for c in selected], dtype=np.float64)
            if np.sum(scores) > 0:
                weights = scores / np.sum(scores)
            else:
                weights = np.full(m_assets, 1.0 / m_assets)
        else:
            weights = f_star / np.sum(f_star)

        # 6. Global Cash Pool & Dual Constraint Sizing (Settled Cash Protected)
        effective_cash = available_cash
        if settled_cash is not None and settled_cash >= 0.0:
            effective_cash = min(available_cash, float(settled_cash))
        cash_pool = float(effective_cash * 0.95)
        r_eff = self.governor.progressive_tracker.get_effective_risk_pct(market_uptrend=market_uptrend)

        raw_allocated = []
        for idx, c in enumerate(selected):
            w_i = float(weights[idx])
            cap_i = min(w_i * cash_pool, portfolio_equity * 0.25)

            action_scale = 0.5 if c.action == 1 else 1.0
            c_risk = r_eff * action_scale
            if c.evt_var_99 > 0.05:
                c_risk *= 0.65

            dollar_risk = portfolio_equity * c_risk
            stop_dist = max(c.stop_dist, c.price * 0.01)

            shares_risk = dollar_risk / stop_dist
            shares_cap = cap_i / c.price if c.price > 0 else 0.0
            target_shares = min(shares_risk, shares_cap)
            stop_loss_price = max(0.01, c.price - stop_dist)

            raw_allocated.append({
                "candidate": c,
                "target_shares": target_shares,
                "stop_loss_price": stop_loss_price,
                "weight": w_i
            })

        # Global Cash Ceiling Invariant: sum(Shares_i * Price_i) <= Cash_Pool
        total_needed = sum(item["target_shares"] * item["candidate"].price for item in raw_allocated)
        if total_needed > cash_pool and total_needed > 0:
            scale_factor = cash_pool / total_needed
            for item in raw_allocated:
                item["target_shares"] *= scale_factor

        total_capital = 0.0
        selected_tickers = [c.ticker for c in selected]

        for item in raw_allocated:
            c = item["candidate"]
            s = item["target_shares"]
            w = item["weight"]
            weights_dict[c.ticker] = round(w, 4)

            if allow_fractional or portfolio_equity < 10000.0 or s < 1.0:
                shares = round(s, 3)
                if shares < 0.001:
                    shares = 0.0
            else:
                shares = int(s)

            if shares <= 0:
                rej_directives.append(OrderDirective(
                    ticker=c.ticker,
                    side=OrderSide.HOLD,
                    shares=0,
                    limit_price=c.price,
                    stop_loss_price=round(item["stop_loss_price"], 2),
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=False,
                    rejection_reason="Share sizing rounded to 0 under capital/cash constraints"
                ))
                continue

            dollar_alloc = round(float(shares * c.price), 2)
            if total_capital + dollar_alloc > cash_pool:
                remaining_cash = max(0.0, cash_pool - total_capital)
                if allow_fractional:
                    shares = round(remaining_cash / c.price, 3)
                    if shares < 0.001:
                        shares = 0.0
                else:
                    shares = int(remaining_cash / c.price)
                dollar_alloc = round(float(shares * c.price), 2)
                if shares <= 0:
                    rej_directives.append(OrderDirective(
                        ticker=c.ticker,
                        side=OrderSide.HOLD,
                        shares=0,
                        limit_price=c.price,
                        stop_loss_price=round(item["stop_loss_price"], 2),
                        dollar_allocation=0.0,
                        risk_pct_actual=0.0,
                        is_authorized=False,
                        rejection_reason="Capital pool exhausted by higher-conviction assets"
                    ))
                    continue

            actual_risk_pct = round(float((shares * (c.price - item["stop_loss_price"])) / portfolio_equity), 4) if portfolio_equity > 0 else 0.0

            child_slices = self.governor.slicer.slice_order(
                shares,
                c.price,
                adv=self.governor.cfg.min_avg_daily_volume_usd,
                cvd_divergence=c.cvd_divergence,
                side=OrderSide.BUY
            )

            self.governor.otr_tracker.record_order(price=c.price, ltp=c.price)

            auth_directives.append(OrderDirective(
                ticker=c.ticker,
                side=OrderSide.BUY,
                shares=shares,
                limit_price=c.price,
                stop_loss_price=round(item["stop_loss_price"], 2),
                dollar_allocation=dollar_alloc,
                risk_pct_actual=actual_risk_pct,
                is_authorized=True,
                rejection_reason=None,
                child_slices=child_slices
            ))
            total_capital += dollar_alloc

        return BatchAllocationResult(
            authorized_directives=auth_directives,
            rejected_directives=rej_directives,
            total_allocated_capital=round(total_capital, 2),
            cash_pool=round(cash_pool, 2),
            weights=weights_dict,
            selected_tickers=selected_tickers
        )


class SyntheticStopLossMonitor:
    """Multi-tick wick-resistant synthetic stop-loss monitor for fractional positions.
    
    Protects micro-accounts from predatory HFT liquidity sweeps, anomalous exchange prints,
    and single-tick flash whipsaws by requiring persistent breach confirmation.
    """
    def __init__(self, confirmation_ticks: int = 3):
        self.confirmation_ticks = confirmation_ticks
        self._positions: dict[str, dict] = {}

    def register_position(self, ticker: str, entry_price: float, stop_loss_price: float) -> None:
        self._positions[ticker] = {
            "entry_price": float(entry_price),
            "stop_loss_price": float(stop_loss_price),
            "breaches": 0,
            "highest_price": float(entry_price),
            "triggered": False
        }

    def unregister_position(self, ticker: str) -> None:
        self._positions.pop(ticker, None)

    def update_stop_loss(self, ticker: str, new_stop_price: float) -> None:
        if ticker in self._positions:
            self._positions[ticker]["stop_loss_price"] = max(self._positions[ticker]["stop_loss_price"], float(new_stop_price))

    def evaluate_tick(self, ticker: str, current_price: float) -> tuple[bool, str | None]:
        """Evaluates price tick against synthetic stop loss with multi-tick persistence.
        
        Returns (is_triggered, reason).
        """
        pos = self._positions.get(ticker)
        if not pos or pos.get("triggered", False):
            return False, None

        stop_loss = pos["stop_loss_price"]
        price = float(current_price)
        pos["highest_price"] = max(pos["highest_price"], price)

        if price < stop_loss:
            pos["breaches"] += 1
            if pos["breaches"] >= self.confirmation_ticks:
                pos["triggered"] = True
                return True, f"Synthetic stop-loss confirmed: {pos['breaches']} consecutive ticks below ${stop_loss:.2f} (current ${price:.2f})"
        else:
            # Price recovered above stop loss -> reset breach count (HFT wick immunity)
            pos["breaches"] = 0

        return False, None


@dataclass
class BarrierState:
    ticker: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    max_bars: int
    bars_held: int = 0
    consecutive_stop_breaches: int = 0
    highest_price: float = 0.0


class TripleBarrierTracker:
    """Marcos López de Prado Triple Barrier Exit Engine.
    
    Tracks:
    1. Upper Barrier: Dynamic Take-Profit (+2.5R)
    2. Lower Barrier: Synthetic Stop-Loss (-1.0R with multi-tick wick filter)
    3. Vertical Barrier: Time-decay expiration (T_max bars) to recycle stagnant capital.
    """
    def __init__(self, default_max_bars: int = 15, wick_confirmation_ticks: int = 3):
        self.default_max_bars = default_max_bars
        self.wick_confirmation_ticks = wick_confirmation_ticks
        self._barriers: dict[str, BarrierState] = {}

    def register(
        self,
        ticker: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float | None = None,
        max_bars: int | None = None
    ) -> None:
        stop_dist = max(0.01, entry_price - stop_loss_price)
        tp = take_profit_price if take_profit_price is not None else (entry_price + 2.5 * stop_dist)
        bars = max_bars if max_bars is not None else self.default_max_bars
        self._barriers[ticker] = BarrierState(
            ticker=ticker,
            entry_price=float(entry_price),
            stop_loss_price=float(stop_loss_price),
            take_profit_price=float(tp),
            max_bars=bars,
            bars_held=0,
            consecutive_stop_breaches=0,
            highest_price=float(entry_price)
        )

    def unregister(self, ticker: str) -> None:
        self._barriers.pop(ticker, None)

    def evaluate_bar(self, ticker: str, current_price: float) -> tuple[bool, str | None]:
        """Evaluates price and holding duration against the 3 barriers.
        
        Returns (should_exit, exit_reason).
        """
        state = self._barriers.get(ticker)
        if not state:
            return False, None

        state.bars_held += 1
        state.highest_price = max(state.highest_price, current_price)

        # 1. Upper Barrier (Take Profit)
        if current_price >= state.take_profit_price:
            return True, f"Triple Barrier: UPPER_PROFIT_BARRIER reached at ${current_price:.2f} >= ${state.take_profit_price:.2f}"

        # 2. Lower Barrier (Stop Loss with Wick filter)
        if current_price < state.stop_loss_price:
            state.consecutive_stop_breaches += 1
            if state.consecutive_stop_breaches >= self.wick_confirmation_ticks:
                return True, f"Triple Barrier: LOWER_STOP_BARRIER breached ({state.consecutive_stop_breaches} ticks < ${state.stop_loss_price:.2f})"
        else:
            state.consecutive_stop_breaches = 0

        # 3. Vertical Barrier (Time Expiration)
        if state.bars_held >= state.max_bars:
            return True, f"Triple Barrier: VERTICAL_TIME_BARRIER expired ({state.bars_held}/{state.max_bars} bars without resolution)"

        return False, None


class RiskGovernor:
    """Mathematical gatekeeper authorizing, sizing, or rejecting trade orders."""

    def __init__(self, risk_cfg: RiskConfig):
        self.cfg = risk_cfg
        self.otr_tracker = OTRTracker(
            soft_limit=getattr(risk_cfg, "otr_soft_threshold", 35.0),
            kill_limit=getattr(risk_cfg, "otr_kill_threshold", 45.0),
            window_sec=getattr(risk_cfg, "otr_window_seconds", 86400),
            proximity_delta=getattr(risk_cfg, "otr_proximity_delta", 0.0075)
        )
        self.slicer = AlmgrenChrissSlicer()
        self.progressive_tracker = ProgressiveExposureTracker(
            base_risk_pct=getattr(risk_cfg, "base_risk_per_trade", 0.005)
        )
        self.allocator = PortfolioCapitalAllocator(self)
        self.synthetic_stop_monitor = SyntheticStopLossMonitor(
            confirmation_ticks=getattr(risk_cfg, "synthetic_stop_confirmation_ticks", 3)
        )
        self.triple_barrier_tracker = TripleBarrierTracker(
            default_max_bars=getattr(risk_cfg, "triple_barrier_max_bars", 15),
            wick_confirmation_ticks=getattr(risk_cfg, "synthetic_stop_confirmation_ticks", 3)
        )

    def evaluate_batch(
        self,
        candidates: list[CandidateIntent],
        portfolio_equity: float,
        available_cash: float,
        current_positions_count: int = 0,
        held_tickers: set[str] | list[str] | None = None,
        circuit_breaker_active: bool = False,
        market_uptrend: bool = True,
        settled_cash: float | None = None
    ) -> BatchAllocationResult:
        """Batch evaluate candidate intents with Multi-Asset Kelly & cash conservation."""
        allow_fractional = getattr(self.cfg, "allow_fractional_shares", True)
        return self.allocator.allocate_batch_directives(
            candidates=candidates,
            portfolio_equity=portfolio_equity,
            available_cash=available_cash,
            current_positions_count=current_positions_count,
            held_tickers=held_tickers,
            circuit_breaker_active=circuit_breaker_active,
            market_uptrend=market_uptrend,
            allow_fractional=allow_fractional,
            settled_cash=settled_cash
        )

    def record_trade_outcome(self, pnl: float, r_multiple: float) -> None:
        """Feeds a completed trade outcome to the progressive exposure tracker."""
        self.progressive_tracker.record_trade_outcome(pnl=pnl, r_multiple=r_multiple)

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
        returns_history: np.ndarray | None = None,
        conformal_residuals: np.ndarray | None = None,
        cvd_divergence: float = 0.0
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
            self.otr_tracker.record_order(price=current_price, ltp=current_price)
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
        allow_fractional = getattr(self.cfg, "allow_fractional_shares", True)
        if allow_fractional and (portfolio_equity < 10000.0 or (dollar_risk / stop_dist) < 1.0):
            raw_shares = round(dollar_risk / stop_dist, 3)
            if raw_shares < 0.001:
                raw_shares = 0.0
        else:
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

        # Fractional Kelly Capital Allocation (with Conformal Uncertainty)
        kelly_fraction = getattr(self.cfg, "kelly_fraction", 0.15)
        max_lev = getattr(self.cfg, "max_leverage", 1.5)
        kelly_capital = calculate_fractional_kelly_capital(
            returns=returns_history,
            portfolio_equity=portfolio_equity,
            kappa=kelly_fraction,
            max_leverage=max_lev,
            max_capital_pct=0.20,
            conformal_residuals=conformal_residuals,
            conformal_alpha=getattr(self.cfg, "conformal_alpha", 0.20),
            conformal_z_score=getattr(self.cfg, "conformal_z_score", 1.2816)
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

        if allow_fractional and (portfolio_equity < 10000.0 or (max_capital_for_pos / current_price) < 1.0):
            max_shares_capital = round(max_capital_for_pos / current_price, 3) if current_price > 0 else 0.0
        else:
            max_shares_capital = int(max_capital_for_pos / current_price) if current_price > 0 else 0

        final_shares = min(raw_shares, max_shares_capital)
        if isinstance(final_shares, float):
            final_shares = round(final_shares, 3)

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

        # Almgren-Chriss Order Slicing for parent orders (modulated by CVD absorption)
        child_slices = self.slicer.slice_order(
            final_shares,
            current_price,
            adv=self.cfg.min_avg_daily_volume_usd,
            cvd_divergence=cvd_divergence,
            side=OrderSide.BUY
        )

        # Record authorized order in OTR tracker (proximity-exempt at market price)
        self.otr_tracker.record_order(price=current_price, ltp=current_price)

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
