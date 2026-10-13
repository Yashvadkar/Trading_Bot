"""Unit tests for Deterministic Order Idempotency and Timeout Reconciliation.

Verifies:
1. client_order_id token generation in all orders
2. Deterministic timeout reconciliation via by_client_order_id
3. Replay-safe handling without double submission
"""

import unittest
from unittest.mock import MagicMock, patch
import requests
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, load_config
from risk.governor import OrderDirective, OrderSide
from broker.alpaca_client import AlpacaPaperClient, ExecutionResult


class TestAlpacaIdempotency(unittest.TestCase):

    def setUp(self):
        cfg = load_config()
        self.client = AlpacaPaperClient(cfg, dry_run=False)

    def test_dry_run_idempotency_token(self):
        dry_client = AlpacaPaperClient(load_config(), dry_run=True)
        directive = OrderDirective(
            ticker="NVDA",
            side=OrderSide.BUY,
            shares=10,
            limit_price=100.0,
            stop_loss_price=95.0,
            dollar_allocation=1000.0,
            risk_pct_actual=0.01,
            is_authorized=True
        )
        res = dry_client.submit_order(directive)
        self.assertTrue(res.success)
        self.assertIn("SIM_NVDA_10_", res.order_id)
        self.assertIn("idempotency key: bot_", res.message)

    @patch("requests.Session.post")
    @patch.object(AlpacaPaperClient, "get_order_by_client_id")
    def test_timeout_reconciliation_order_accepted(self, mock_get_order, mock_post):
        # Simulate network timeout during initial POST
        mock_post.side_effect = requests.Timeout("Network timed out waiting for gateway")
        
        # Simulate exchange confirms the order was actually accepted
        mock_get_order.return_value = {
            "id": "alpaca_order_reconciled_999",
            "status": "accepted",
            "client_order_id": "bot_test123"
        }

        directive = OrderDirective(
            ticker="AAPL",
            side=OrderSide.BUY,
            shares=25,
            limit_price=150.0,
            stop_loss_price=145.0,
            dollar_allocation=3750.0,
            risk_pct_actual=0.015,
            is_authorized=True
        )

        res = self.client.submit_order(directive)
        self.assertTrue(res.success)
        self.assertEqual(res.order_id, "alpaca_order_reconciled_999")
        self.assertIn("Order reconciled via client_order_id", res.message)

    @patch("requests.Session.post")
    @patch.object(AlpacaPaperClient, "get_order_by_client_id")
    def test_timeout_reconciliation_order_unconfirmed(self, mock_get_order, mock_post):
        # Simulate network timeout and exchange has no record of the order
        mock_post.side_effect = requests.Timeout("Network timed out")
        mock_get_order.return_value = None

        directive = OrderDirective(
            ticker="AAPL",
            side=OrderSide.BUY,
            shares=25,
            limit_price=150.0,
            stop_loss_price=145.0,
            dollar_allocation=3750.0,
            risk_pct_actual=0.015,
            is_authorized=True
        )

        res = self.client.submit_order(directive)
        self.assertFalse(res.success)
        self.assertIsNone(res.order_id)
        self.assertIn("Order timed out and was not confirmed", res.message)

    def test_pre_trade_spread_guard_rejection(self):
        # Bid = $99.00, Ask = $101.00 -> Spread = $2.00 / $100.00 = 2.0% > 0.30%
        ok, spread, reason = self.client.check_pre_trade_spread("WIDE_STOCK", bid=99.0, ask=101.0)
        self.assertFalse(ok)
        self.assertAlmostEqual(spread, 0.02, places=4)
        self.assertIn("exceeds max allowable threshold", reason)

    def test_pre_trade_spread_guard_accepted(self):
        # Bid = $100.00, Ask = $100.10 -> Spread = $0.10 / $100.05 = 0.099% < 0.30%
        ok, spread, reason = self.client.check_pre_trade_spread("TIGHT_STOCK", bid=100.0, ask=100.10)
        self.assertTrue(ok)
        self.assertLess(spread, 0.003)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()

