"""Unit tests for Multi-Asset Covariance Kelly & Small-Account Portfolio Allocator.

Verifies:
1. Small-account cardinality constraints (K <= 2 for < $5k, K <= 3 for < $10k).
2. Global Cash Ceiling Invariant: total allocated capital <= available_cash * 0.95.
3. Conviction ranking based on debate consensus and Sortino ratio.
4. Multi-Asset Covariance Kelly Vector allocation across correlated assets.
5. Early EXIT directives pass through as authorized SELL orders.
6. Systemic circuit breaker blocks all BUY allocations.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RiskConfig
from risk.governor import (
    RiskGovernor, CandidateIntent, OrderSide, PortfolioCapitalAllocator
)


class TestPortfolioAllocator(unittest.TestCase):

    def setUp(self):
        self.cfg = RiskConfig(
            max_concurrent_positions=5,
            max_risk_per_trade=0.015,
            kelly_fraction=0.15,
            min_avg_daily_volume_usd=10_000_000.0
        )
        self.governor = RiskGovernor(self.cfg)
        self.allocator = self.governor.allocator

    def test_small_account_cardinality_under_5k(self):
        # Under $5,000 equity, cardinality must be strictly capped at K = 2
        candidates = [
            CandidateIntent(ticker="NVDA", price=100.0, atr=2.0, stop_dist=4.0, debate_confidence=0.9, sortino_ratio=2.5),
            CandidateIntent(ticker="AAPL", price=150.0, atr=3.0, stop_dist=6.0, debate_confidence=0.8, sortino_ratio=2.0),
            CandidateIntent(ticker="MSFT", price=200.0, atr=4.0, stop_dist=8.0, debate_confidence=0.7, sortino_ratio=1.8),
            CandidateIntent(ticker="AMD", price=80.0, atr=2.0, stop_dist=4.0, debate_confidence=0.6, sortino_ratio=1.5),
            CandidateIntent(ticker="TSLA", price=180.0, atr=5.0, stop_dist=10.0, debate_confidence=0.5, sortino_ratio=1.0),
        ]

        result = self.allocator.allocate_batch_directives(
            candidates=candidates,
            portfolio_equity=3000.0,
            available_cash=2500.0,
            current_positions_count=0
        )

        # Top 2 by conviction are NVDA and AAPL
        self.assertEqual(len(result.authorized_directives), 2)
        auth_tickers = [d.ticker for d in result.authorized_directives]
        self.assertEqual(auth_tickers, ["NVDA", "AAPL"])

        # Remaining 3 must be in rejected_directives
        rej_tickers = [d.ticker for d in result.rejected_directives]
        self.assertIn("MSFT", rej_tickers)
        self.assertIn("AMD", rej_tickers)
        self.assertIn("TSLA", rej_tickers)

    def test_small_account_cardinality_between_5k_and_10k(self):
        # Between $5,000 and $10,000 equity, cardinality must be K = 3
        candidates = [
            CandidateIntent(ticker="NVDA", price=100.0, atr=2.0, stop_dist=4.0, debate_confidence=0.9, sortino_ratio=2.5),
            CandidateIntent(ticker="AAPL", price=150.0, atr=3.0, stop_dist=6.0, debate_confidence=0.8, sortino_ratio=2.0),
            CandidateIntent(ticker="MSFT", price=200.0, atr=4.0, stop_dist=8.0, debate_confidence=0.7, sortino_ratio=1.8),
            CandidateIntent(ticker="AMD", price=80.0, atr=2.0, stop_dist=4.0, debate_confidence=0.6, sortino_ratio=1.5),
        ]

        result = self.allocator.allocate_batch_directives(
            candidates=candidates,
            portfolio_equity=7500.0,
            available_cash=6000.0,
            current_positions_count=0
        )

        self.assertEqual(len(result.authorized_directives), 3)
        auth_tickers = [d.ticker for d in result.authorized_directives]
        self.assertEqual(auth_tickers, ["NVDA", "AAPL", "MSFT"])

    def test_zero_cash_overdraw_invariant(self):
        # Available cash = $1,000. Cash pool = $950.00
        # Candidates request large position sizes
        candidates = [
            CandidateIntent(ticker="A", price=50.0, atr=1.0, stop_dist=2.0, debate_confidence=0.9, sortino_ratio=2.0),
            CandidateIntent(ticker="B", price=100.0, atr=2.0, stop_dist=3.0, debate_confidence=0.8, sortino_ratio=1.8),
        ]

        result = self.allocator.allocate_batch_directives(
            candidates=candidates,
            portfolio_equity=2000.0,
            available_cash=1000.0,
            current_positions_count=0
        )

        cash_pool = 1000.0 * 0.95  # $950.00
        self.assertLessEqual(result.total_allocated_capital, cash_pool)
        self.assertGreater(result.total_allocated_capital, 0.0)

        # Sum of individual authorized allocations must not exceed cash_pool
        sum_alloc = sum(d.dollar_allocation for d in result.authorized_directives)
        self.assertLessEqual(sum_alloc, cash_pool)

    def test_multi_asset_covariance_kelly_with_returns(self):
        # Generate correlated synthetic return series
        np.random.seed(42)
        r_a = np.random.normal(0.005, 0.02, 30)
        r_b = r_a * 0.8 + np.random.normal(0.002, 0.01, 30)  # High correlation

        candidates = [
            CandidateIntent(ticker="STOCK_A", price=100.0, atr=2.0, stop_dist=4.0, returns=r_a, debate_confidence=0.8),
            CandidateIntent(ticker="STOCK_B", price=100.0, atr=2.0, stop_dist=4.0, returns=r_b, debate_confidence=0.7),
        ]

        result = self.allocator.allocate_batch_directives(
            candidates=candidates,
            portfolio_equity=50_000.0,
            available_cash=40_000.0,
            current_positions_count=0
        )

        self.assertEqual(len(result.authorized_directives), 2)
        # Weights should sum to 1.0
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=3)
        self.assertLessEqual(result.total_allocated_capital, 40_000.0 * 0.95)

    def test_exit_action_and_circuit_breaker(self):
        # Action 3: EXIT signal
        exit_cand = CandidateIntent(ticker="HELD_EXIT", price=120.0, atr=2.0, stop_dist=0.0, action=3)
        buy_cand = CandidateIntent(ticker="NEW_BUY", price=50.0, atr=1.0, stop_dist=2.0, action=2)

        # Under circuit breaker active:
        res_cb = self.allocator.allocate_batch_directives(
            candidates=[buy_cand],
            portfolio_equity=10_000.0,
            available_cash=8_000.0,
            circuit_breaker_active=True
        )
        self.assertEqual(len(res_cb.authorized_directives), 0)
        self.assertIn("Circuit breaker active", res_cb.rejected_directives[0].rejection_reason)

        # Normal run with exit directive:
        res_exit = self.allocator.allocate_batch_directives(
            candidates=[exit_cand, buy_cand],
            portfolio_equity=10_000.0,
            available_cash=8_000.0,
            circuit_breaker_active=False
        )
        auth_sides = {d.ticker: d.side for d in res_exit.authorized_directives}
        self.assertEqual(auth_sides["HELD_EXIT"], OrderSide.SELL)
        self.assertEqual(auth_sides["NEW_BUY"], OrderSide.BUY)


if __name__ == "__main__":
    unittest.main()
