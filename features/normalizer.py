"""Multi-Branch 64-Dimensional Quantitative State Normalizer.

Transforms heterogeneous mathematical physics, dynamical systems, and financial signals
into a fixed 64-dimensional normalized PyTorch FloatTensor for the neural ensemble:

- Branch 1 (Dims 0-11, 12 features): Rough Path Signatures (Level-1 increments & Level-2 Lévy Area)
- Branch 2 (Dims 12-21, 10 features): Market Field & RMT (Denoised eigenvalues, Transfer Entropy, MST Centrality)
- Branch 3 (Dims 22-31, 10 features): Dynamical Chaos & EVT (Hurst exponent, Critical Slowing Down, GPD VaR/ES)
- Branch 4 (Dims 32-47, 16 features): ELM Liquid Reservoir (High-dimensional non-linear liquid state)
- Branch 5 (Dims 48-55, 8 features): Fundamentals & Microstructure (Valuation, balance sheet, order book spread)
- Branch 6 (Dims 56-63, 8 features): Portfolio Context & Debate Dynamics (Allocation, drawdown, epistemic uncertainty)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
from features.indicators import TechnicalMetrics
from features.quant_theories import QuantTheoryEngine, QuantTheoryFeatures


@dataclass
class SentimentMetrics:
    """Live online catalyst metrics (decoupled from pure quantitative historical pretraining)."""
    mention_count: int = 5
    sentiment_score: float = 0.0          # -1.0 (strongly bearish) to +1.0 (strongly bullish)
    catalyst_urgency: float = 0.5         # 0.0 to 1.0 (e.g. earnings today vs general chatter)
    source_credibility: float = 0.8       # 0.0 to 1.0
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
    """Transforms multi-physics inputs into normalized [-3.0, 3.0] 64-dim FloatTensor."""

    def __init__(self, seed: int = 42):
        self.quant_engine = QuantTheoryEngine(reservoir_dim=16, seed=seed)

    def build_state_vector(
        self,
        tech: TechnicalMetrics,
        sent: SentimentMetrics | None = None,
        fund: FundamentalMetrics | None = None,
        micro: MicrostructureMetrics | None = None,
        port: PortfolioContext | None = None,
        debate: DebateMetrics | None = None,
        quant: QuantTheoryFeatures | None = None
    ) -> torch.FloatTensor:
        """Assembles exactly 64 bounded features into a PyTorch FloatTensor of shape (1, 64)."""
        sent = sent or SentimentMetrics()
        fund = fund or FundamentalMetrics()
        micro = micro or MicrostructureMetrics()
        port = port or PortfolioContext()
        debate = debate or DebateMetrics()

        # If quant theories are not passed directly, generate physically sound proxies from tech indicators
        if quant is None:
            # Synthetic 20-bar trajectory reconstructed from technical parameters
            p_end = 100.0
            p_start = p_end / (tech.sma_20_ratio + 1e-4)
            sim_prices = np.linspace(p_start, p_end, 20)
            sim_vols = np.ones(20) * 1e6
            quant = self.quant_engine.extract_all_features(sim_prices, sim_vols)

        features = np.zeros(64, dtype=np.float32)

        # ---------------------------------------------------------------------
        # Branch 1: Rough Path Signatures (12 features, Dims 0-11)
        # ---------------------------------------------------------------------
        sig = quant.signature_tensor[:12]
        features[0:12] = np.clip(sig, -3.0, 3.0)

        # ---------------------------------------------------------------------
        # Branch 2: Market Field & RMT (10 features, Dims 12-21)
        # ---------------------------------------------------------------------
        mkt = quant.market_field_vector[:10]
        features[12:22] = np.clip(mkt, -3.0, 3.0)

        # ---------------------------------------------------------------------
        # Branch 3: Dynamical Chaos & EVT (10 features, Dims 22-31)
        # ---------------------------------------------------------------------
        chaos = quant.chaos_evt_vector[:10]
        features[22:32] = np.clip(chaos, -3.0, 3.0)

        # ---------------------------------------------------------------------
        # Branch 4: ELM Liquid Reservoir (16 features, Dims 32-47)
        # ---------------------------------------------------------------------
        elm = quant.elm_latent_state[:16]
        features[32:48] = np.clip(elm, -3.0, 3.0)

        # ---------------------------------------------------------------------
        # Branch 5: Fundamentals & Microstructure (8 features, Dims 48-55)
        # ---------------------------------------------------------------------
        log_mcap = np.log10(max(fund.market_cap, 1e6)) - 9.0  # Centered at $1B
        features[48] = float(np.clip(log_mcap / 2.0, -2.0, 3.0))
        features[49] = float(np.clip((fund.pe_ratio - 20.0) / 15.0, -2.0, 3.0))
        features[50] = float(np.clip((fund.debt_to_equity - 1.0) / 1.5, -2.0, 3.0))
        features[51] = float(np.clip(fund.fcf_margin * 5.0, -2.0, 3.0))
        features[52] = float(np.clip(fund.revenue_growth_yoy * 5.0, -2.0, 3.0))
        features[53] = float(np.clip((fund.beta - 1.0) * 2.0, -2.0, 2.0))
        features[54] = float(np.clip(micro.spread_pct * 1000.0, 0.0, 3.0))
        features[55] = float(np.clip((micro.volume_surge_ratio - 1.0) * 1.5, -2.0, 3.0))

        # ---------------------------------------------------------------------
        # Branch 6: Portfolio Context & Debate Dynamics (8 features, Dims 56-63)
        # ---------------------------------------------------------------------
        features[56] = float(np.clip((port.cash_ratio - 0.5) * 2.0, -1.0, 1.0))
        features[57] = float(np.clip(port.current_stock_weight * 10.0, 0.0, 2.0))
        features[58] = float(np.clip(port.unrealized_pnl_pct * 10.0, -3.0, 3.0))
        features[59] = float(np.clip(port.daily_drawdown_pct * 20.0, 0.0, 3.0))
        features[60] = float(np.clip((debate.bull_confidence - 0.5) * 2.0, -1.0, 1.0))
        features[61] = float(np.clip((debate.bear_confidence - 0.5) * 2.0, -1.0, 1.0))
        features[62] = float(np.clip(debate.disagreement_delta * 2.0, -2.0, 2.0))
        features[63] = float(np.clip(debate.disagreement_variance * 4.0, 0.0, 3.0))

        # Numerical safety: sanitize NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

        tensor = torch.from_numpy(features).unsqueeze(0)  # Shape (1, 64)
        return tensor
