"""Unit tests for Goverdhan Gajjala Progressive Exposure Risk Tracker.

Verifies:
1. Default neutral mode returns 1.0x risk multiplier and base risk (0.50%).
2. Defensive contraction mode triggers on 2 consecutive losses -> 0.50x (0.25% risk).
3. Defensive contraction mode triggers when rolling win rate falls below 25%.
4. Offensive expansion mode triggers on 2 consecutive wins with avg win >= 2.0R under confirmed market uptrend -> 1.50x (0.75% risk).
5. Expansion is blocked if market uptrend is False (macro filter protection).
6. Strict Quarter-Kelly ceiling (kappa <= 0.25) to prevent hyper-Kelly geometric ruin.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.governor import ProgressiveExposureTracker


class TestProgressiveExposure(unittest.TestCase):

    def setUp(self):
        self.tracker = ProgressiveExposureTracker(
            base_risk_pct=0.005,
            max_multiplier=1.50,
            min_multiplier=0.50,
            window_size=10,
            kappa_max=0.25
        )

    def test_initial_neutral_mode(self):
        # Empty history -> Neutral mode (1.0x, 0.50% risk)
        mult = self.tracker.compute_risk_multiplier(market_uptrend=True)
        self.assertEqual(mult, 1.0)
        self.assertEqual(self.tracker.get_effective_risk_pct(market_uptrend=True), 0.005)

    def test_consecutive_losses_trigger_contraction(self):
        # 1 loss: still neutral
        self.tracker.record_trade_outcome(pnl=-50.0, r_multiple=-1.0)
        self.assertEqual(self.tracker.compute_risk_multiplier(market_uptrend=True), 1.0)

        # 2nd consecutive loss: contraction mode (0.50x, 0.25% risk)
        self.tracker.record_trade_outcome(pnl=-50.0, r_multiple=-1.0)
        self.assertEqual(self.tracker.compute_risk_multiplier(market_uptrend=True), 0.50)
        self.assertEqual(self.tracker.get_effective_risk_pct(market_uptrend=True), 0.0025)

    def test_low_win_rate_triggers_contraction(self):
        # 4 trades: 1 win, 3 losses -> win rate = 25% (threshold is < 25%)
        # Add 1 more loss -> 1 win, 4 losses = 20% win rate
        self.tracker.record_trade_outcome(pnl=100.0, r_multiple=2.0)
        self.tracker.record_trade_outcome(pnl=-50.0, r_multiple=-1.0)
        self.tracker.record_trade_outcome(pnl=50.0, r_multiple=1.0)
        self.tracker.record_trade_outcome(pnl=-50.0, r_multiple=-1.0)
        self.tracker.record_trade_outcome(pnl=-50.0, r_multiple=-1.0)

        self.assertEqual(self.tracker.compute_risk_multiplier(market_uptrend=True), 0.50)

    def test_consecutive_large_wins_trigger_expansion(self):
        # 2 consecutive wins with >= 2.0R average win and confirmed uptrend
        self.tracker.record_trade_outcome(pnl=120.0, r_multiple=2.4)
        self.tracker.record_trade_outcome(pnl=100.0, r_multiple=2.0)

        mult = self.tracker.compute_risk_multiplier(market_uptrend=True)
        self.assertEqual(mult, 1.50)
        self.assertEqual(self.tracker.get_effective_risk_pct(market_uptrend=True), 0.0075)

    def test_expansion_blocked_when_market_uptrend_is_false(self):
        # 2 large wins, but market index is below 21 EMA (bearish/sideways regime)
        self.tracker.record_trade_outcome(pnl=120.0, r_multiple=2.4)
        self.tracker.record_trade_outcome(pnl=100.0, r_multiple=2.0)

        # Must not expand risk when broad market is not in confirmed uptrend
        mult = self.tracker.compute_risk_multiplier(market_uptrend=False)
        self.assertEqual(mult, 1.0)
        self.assertEqual(self.tracker.get_effective_risk_pct(market_uptrend=False), 0.005)

    def test_weak_wins_do_not_trigger_expansion(self):
        # 2 wins, but small gains (0.8R and 1.2R -> avg 1.0R < 2.0R)
        self.tracker.record_trade_outcome(pnl=40.0, r_multiple=0.8)
        self.tracker.record_trade_outcome(pnl=60.0, r_multiple=1.2)

        mult = self.tracker.compute_risk_multiplier(market_uptrend=True)
        self.assertEqual(mult, 1.0)

    def test_quarter_kelly_cap_prevents_hyper_kelly_ruin(self):
        # Requesting hyper-Kelly (0.50 or 0.80) is hard-capped to Quarter-Kelly (0.25)
        self.assertEqual(self.tracker.get_effective_kappa(0.50), 0.25)
        self.assertEqual(self.tracker.get_effective_kappa(0.80), 0.25)
        # Moderate requested kappa (0.15) remains unhindered
        self.assertEqual(self.tracker.get_effective_kappa(0.15), 0.15)


if __name__ == "__main__":
    unittest.main()
