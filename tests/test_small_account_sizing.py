"""Unit tests for Small-Account Viability, Fractional Shares, and Retail Bypass.

Verifies:
1. Micro-account ($1,000) sizing on high-priced equities ($400/share) does not round to 0.
2. Fractional shares correctly size with 3 decimal places (0.001 share precision).
3. Almgren-Chriss retail bypass gate returns single block for small orders (< 1% of 5-min ADV).
4. Large institutional orders exceeding ADV threshold continue to be sliced into child blocks.
5. AlpacaPaperClient dry-run successfully processes fractional share directives.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, RiskConfig, ModelConfig, ScoutConfig
from risk.governor import RiskGovernor, AlmgrenChrissSlicer, OrderSide
from broker.alpaca_client import AlpacaPaperClient


class TestSmallAccountSizing(unittest.TestCase):

    def setUp(self):
        self.risk_cfg = RiskConfig(
            max_concurrent_positions=2,
            max_risk_per_trade=0.015,
            min_avg_daily_volume_usd=50_000_000.0
        )
        self.governor = RiskGovernor(self.risk_cfg)
        self.config = Config(
            alpaca_api_key="TEST_KEY",
            alpaca_secret_key="TEST_SECRET",
            alpaca_base_url="https://paper-api.alpaca.markets",
            paper_mode=True,
            risk=self.risk_cfg,
            model=ModelConfig(),
            scout=ScoutConfig()
        )
        self.broker = AlpacaPaperClient(self.config, dry_run=True)

    def test_small_account_fractional_share_sizing(self):
        # Account equity = $1,000, Stock Price = $400 (e.g. MSFT/SPY)
        # Max 20% capital = $200. Max risk = $1,000 * 0.015 = $15.
        # ATR stop dist = $20. Shares by risk = 15 / 20 = 0.75 shares.
        # Shares by capital = 200 / 400 = 0.50 shares.
        # Integer sizing would round min(0.75, 0.50) down to 0, rejecting the order!
        # Fractional sizing must authorize ~0.500 shares.
        directive = self.governor.evaluate_order(
            ticker="MSFT",
            action=2,
            current_price=400.0,
            atr=10.0,
            portfolio_equity=1000.0,
            available_cash=1000.0,
            current_positions_count=0,
            is_already_held=False,
            circuit_breaker_active=False
        )

        self.assertTrue(directive.is_authorized)
        self.assertIsInstance(directive.shares, float)
        self.assertAlmostEqual(directive.shares, 0.500, places=3)
        self.assertLessEqual(directive.dollar_allocation, 250.0)

    def test_almgren_chriss_retail_bypass(self):
        # ADV = $50,000,000. Price = $100. Daily shares = 500,000.
        # 5-min ADV = 500,000 / 78 = 6,410 shares.
        # 1% of 5-min ADV = 64.1 shares.
        # Order of 50 shares ($5,000) is below retail threshold -> bypass slicing, single block
        slices_retail = AlmgrenChrissSlicer.slice_order(
            total_shares=50,
            price=100.0,
            adv=50_000_000.0,
            threshold_usd=2_000.0  # even if threshold is low
        )
        self.assertEqual(slices_retail, [50])

    def test_almgren_chriss_institutional_slicing_active(self):
        # Order of 2,000 shares ($200,000) >> 1% of 5-min ADV and >> threshold_usd ($10,000)
        # Must be sliced into multiple TWAP/POV blocks
        slices_inst = AlmgrenChrissSlicer.slice_order(
            total_shares=2000,
            price=100.0,
            adv=500_000.0,
            max_pov_pct=0.05,
            threshold_usd=10_000.0
        )
        self.assertGreater(len(slices_inst), 1)
        self.assertEqual(sum(slices_inst), 2000)

    def test_broker_fractional_order_dry_run(self):
        # Directive with 0.35 shares
        directive = self.governor.evaluate_order(
            ticker="NVDA",
            action=2,
            current_price=500.0,
            atr=10.0,
            portfolio_equity=1000.0,
            available_cash=1000.0,
            current_positions_count=0,
            is_already_held=False,
            circuit_breaker_active=False
        )
        self.assertTrue(directive.is_authorized)
        self.assertIsInstance(directive.shares, float)

        # Broker submit_order in dry_run mode
        result = self.broker.submit_order(directive)
        self.assertTrue(result.success)
        self.assertEqual(result.ticker, "NVDA")
        self.assertEqual(result.shares, directive.shares)
        self.assertIn("Simulated order accepted", result.message)


if __name__ == "__main__":
    unittest.main()
