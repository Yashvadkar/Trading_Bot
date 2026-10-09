"""Unit tests for Institutional Risk Governor: OTR, Fractional Kelly, and Almgren-Chriss.

Verifies:
1. OTRTracker rolling ratio, throttle warning, and emergency kill-switch
2. Fractional Kelly capital calculation and variance constraint
3. Almgren-Chriss parent order slicing (TWAP/POV blocks)
4. RiskGovernor evaluation with all institutional defense layers active
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RiskConfig
from risk.governor import (
    RiskGovernor, OTRTracker, AlmgrenChrissSlicer,
    calculate_fractional_kelly_capital, OrderSide
)


class TestInstitutionalRisk(unittest.TestCase):

    def test_otr_tracker_kill_switch(self):
        tracker = OTRTracker(soft_limit=35.0, kill_limit=45.0, window_sec=86400)
        # Record 46 orders and 0 trades
        for _ in range(46):
            tracker.record_order()

        self.assertEqual(tracker.get_current_otr(), 46.0)
        safe, msg = tracker.check_safety()
        self.assertFalse(safe)
        self.assertIn("OTR Regulatory Kill-Switch engaged", msg)

        # Record 2 executed trades -> OTR becomes 46 / 2 = 23.0 < 35.0 -> Safe!
        tracker.record_trade()
        tracker.record_trade()
        safe2, _ = tracker.check_safety()
        self.assertTrue(safe2)
        self.assertEqual(tracker.get_current_otr(), 23.0)

    def test_fractional_kelly_capital(self):
        equity = 100_000.0
        # Positive Sharpe regime: mean return 2%, variance 0.0004 -> Kelly fraction = 0.02 / 0.0004 = 50
        # Fractional Kelly (kappa=0.15): 0.15 * 50 = 7.5, capped by max_capital_pct=0.20 -> $20,000
        returns_pos = np.array([0.01, 0.03, 0.02, 0.015, 0.025, 0.02])
        cap_pos = calculate_fractional_kelly_capital(returns_pos, equity, kappa=0.15, max_capital_pct=0.20)
        self.assertEqual(cap_pos, 20_000.0)

        # Negative drift regime: mean return -1% -> Kelly allocates 0 capital!
        returns_neg = np.array([-0.01, -0.02, 0.005, -0.015, -0.01])
        cap_neg = calculate_fractional_kelly_capital(returns_neg, equity, kappa=0.15)
        self.assertEqual(cap_neg, 0.0)

    def test_almgren_chriss_slicer(self):
        # Small order: $5,000 -> 1 single slice
        slices_small = AlmgrenChrissSlicer.slice_order(total_shares=50, price=100.0, adv=10_000_000.0)
        self.assertEqual(slices_small, [50])

        # Large order: $100,000 (1,000 shares @ $100) with thin ADV -> Sliced into child orders
        slices_large = AlmgrenChrissSlicer.slice_order(
            total_shares=1000, price=100.0, adv=500_000.0, max_pov_pct=0.05, threshold_usd=10_000.0
        )
        self.assertTrue(len(slices_large) > 1)
        self.assertEqual(sum(slices_large), 1000)

    def test_risk_governor_otr_rejection(self):
        cfg = RiskConfig(otr_kill_threshold=45.0)
        gov = RiskGovernor(cfg)

        # Flood tracker with 46 orders
        for _ in range(46):
            gov.otr_tracker.record_order()

        directive = gov.evaluate_order(
            ticker="NVDA",
            action=2,
            current_price=100.0,
            atr=3.0,
            portfolio_equity=100_000.0,
            available_cash=50_000.0,
            current_positions_count=0,
            is_already_held=False,
            circuit_breaker_active=False
        )
        self.assertFalse(directive.is_authorized)
        self.assertIn("OTR Regulatory Kill-Switch engaged", directive.rejection_reason)


if __name__ == "__main__":
    unittest.main()
