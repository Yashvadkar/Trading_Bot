"""State Vector Normalizer for Multi-Neural-Network Debate Architecture.

Transforms heterogeneous financial signals into a fixed 32-dimensional
normalized PyTorch FloatTensor suitable for low-latency GRU / MLP consumption.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
from features.indicators import TechnicalMetrics


@dataclass
class SentimentMetrics:
    mention_count: int = 5
    sentiment_score: float = 0.0          # -1.0 (strongly bearish) to +1.0 (strongly bullish)
    catalyst_urgency: float = 0.5         # 0.0 to 1.0 (e.g. earnings today vs general chatter)
    source_credibility: float = 0.8       # 0.0 to 1.0 (e.g. SEC 8-K = 1.0, Reddit = 0.5)
    sentiment_delta_24h: float = 0.0      # Change in sentiment over 24h
    social_velocity: float = 1.0          # Mentions growth rate vs 7-day average


@dataclass
class FundamentalMetrics:
    market_cap: float = 10_000_000_000.0  # $10B default large cap
    pe_ratio: float = 20.0                # Price-to-Earnings
    debt_to_equity: float = 1.0           # Debt / Equity
    fcf_margin: float = 0.15              # Free cash flow margin (15%)
    revenue_growth_yoy: float = 0.10      # 10% YoY revenue growth
    beta: float = 1.0                     # Market beta


@dataclass
class MicrostructureMetrics:
    spread_pct: float = 0.0005            # Bid-ask spread % (0.05%)
    bid_ask_imbalance: float = 0.0        # -1.0 to 1.0 order book imbalance proxy
    close_to_high_ratio: float = 0.95     # Close / Day High
    volume_surge_ratio: float = 1.0


@dataclass
class PortfolioContext:
    cash_ratio: float = 0.80              # 80% portfolio in cash
    current_stock_weight: float = 0.0     # 0% currently allocated to this ticker
    unrealized_pnl_pct: float = 0.0       # 0% unrealized PnL
    daily_drawdown_pct: float = 0.0       # Current day's portfolio drawdown


@dataclass
class DebateMetrics:
    bull_confidence: float = 0.5          # 0.0 to 1.0
    bear_confidence: float = 0.5          # 0.0 to 1.0
    disagreement_delta: float = 0.0       # Bull - Bear (-1.0 to +1.0)
    disagreement_variance: float = 0.0    # sigma^2 (0.0 to 1.0)


class StateNormalizer:
    """Transforms 32 heterogeneous inputs into normalized [-3.0, 3.0] tensor."""

    def __init__(self):
        pass

    def build_state_vector(
        self,
        tech: TechnicalMetrics,
        sent: SentimentMetrics,
        fund: FundamentalMetrics,
        micro: MicrostructureMetrics,
        port: PortfolioContext,
        debate: DebateMetrics
    ) -> torch.FloatTensor:
        """Assembles and bounds exactly 32 float features."""
        features = np.zeros(32, dtype=np.float32)

        # 1-8: Technical Indicators (8 features)
        features[0] = float(np.clip((tech.rsi - 50.0) / 25.0, -2.0, 2.0))
        features[1] = float(np.clip(tech.macd_hist * 10.0, -3.0, 3.0))
        features[2] = float(np.clip(tech.atr_pct * 25.0, 0.0, 3.0))
        features[3] = float(np.clip(tech.sma_20_ratio * 10.0, -3.0, 3.0))
        features[4] = float(np.clip(tech.sma_50_ratio * 5.0, -3.0, 3.0))
        features[5] = float(np.clip(tech.sma_200_ratio * 3.0, -3.0, 3.0))
        features[6] = float(np.clip((tech.bb_pct_b - 0.5) * 2.0, -2.0, 2.0))
        features[7] = float(np.clip((tech.realized_volatility - 0.20) * 5.0, -2.0, 3.0))

        # 9-14: Sentiment & Discovery (6 features)
        features[8] = float(np.clip(np.log1p(sent.mention_count) / 4.0, 0.0, 3.0))
        features[9] = float(np.clip(sent.sentiment_score * 2.0, -2.0, 2.0))
        features[10] = float(np.clip((sent.catalyst_urgency - 0.5) * 2.0, -1.0, 1.0))
        features[11] = float(np.clip(sent.source_credibility, 0.0, 1.0))
        features[12] = float(np.clip(sent.sentiment_delta_24h * 2.0, -2.0, 2.0))
        features[13] = float(np.clip(np.log1p(sent.social_velocity) - 0.693, -2.0, 3.0))

        # 15-20: Fundamental Health (6 features)
        log_mcap = np.log10(max(fund.market_cap, 1e6)) - 9.0  # Normalized around $1B (10^9)
        features[14] = float(np.clip(log_mcap / 2.0, -2.0, 3.0))
        features[15] = float(np.clip((fund.pe_ratio - 20.0) / 15.0, -2.0, 3.0))
        features[16] = float(np.clip((fund.debt_to_equity - 1.0) / 1.5, -2.0, 3.0))
        features[17] = float(np.clip(fund.fcf_margin * 5.0, -2.0, 3.0))
        features[18] = float(np.clip(fund.revenue_growth_yoy * 5.0, -2.0, 3.0))
        features[19] = float(np.clip((fund.beta - 1.0) * 2.0, -2.0, 2.0))

        # 21-24: Market Microstructure (4 features)
        features[20] = float(np.clip(micro.spread_pct * 1000.0, 0.0, 3.0))
        features[21] = float(np.clip(micro.bid_ask_imbalance, -1.0, 1.0))
        features[22] = float(np.clip((micro.close_to_high_ratio - 0.95) * 20.0, -2.0, 2.0))
        features[23] = float(np.clip((micro.volume_surge_ratio - 1.0) * 1.5, -2.0, 3.0))

        # 25-28: Portfolio Context (4 features)
        features[24] = float(np.clip((port.cash_ratio - 0.5) * 2.0, -1.0, 1.0))
        features[25] = float(np.clip(port.current_stock_weight * 10.0, 0.0, 2.0))
        features[26] = float(np.clip(port.unrealized_pnl_pct * 10.0, -3.0, 3.0))
        features[27] = float(np.clip(port.daily_drawdown_pct * 20.0, 0.0, 3.0))

        # 29-32: Adversarial Debate Metrics (4 features)
        features[28] = float(np.clip((debate.bull_confidence - 0.5) * 2.0, -1.0, 1.0))
        features[29] = float(np.clip((debate.bear_confidence - 0.5) * 2.0, -1.0, 1.0))
        features[30] = float(np.clip(debate.disagreement_delta * 2.0, -2.0, 2.0))
        features[31] = float(np.clip(debate.disagreement_variance * 4.0, 0.0, 3.0))

        # Handle any possible NaN/Inf from invalid inputs
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

        tensor = torch.from_numpy(features).unsqueeze(0)  # Shape (1, 32)
        return tensor
