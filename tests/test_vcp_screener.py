"""Unit tests for Minervini/Gajjala Intraday Volatility Contraction Pattern (VCP) Screener."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.indicators import compute_ema, detect_volatility_contraction_pattern
from screening.screener import QuantitativeScreener, ScreenedStock
from config import RiskConfig


class TestVCPScreener(unittest.TestCase):

    def test_compute_ema(self):
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
        ema = compute_ema(prices, span=3)
        self.assertEqual(len(ema), len(prices))
        self.assertEqual(ema[0], 10.0)
        self.assertGreater(ema[-1], 12.5)

    def test_vcp_positive_pattern(self):
        # Generate 40 bars with contracting ranges and drying volume in an uptrend
        np.random.seed(42)
        n = 40
        # Uptrending price base from 100 to 120
        base_trend = np.linspace(100.0, 120.0, n)
        
        # Wave 1 (first 20 bars): wide swings (+- 10)
        # Wave 2 (last 20 bars): tight swings (+- 2)
        swings = np.concatenate([
            np.sin(np.linspace(0, 3.14 * 2, 20)) * 8.0,
            np.sin(np.linspace(0, 3.14 * 2, 20)) * 1.5
        ])
        closes = base_trend + swings
        highs = closes + np.concatenate([np.full(20, 2.0), np.full(20, 0.5)])
        lows = closes - np.concatenate([np.full(20, 2.0), np.full(20, 0.5)])
        
        # Volumes: high in wave 1, drying up in wave 2
        volumes = np.concatenate([
            np.full(20, 200_000.0),
            np.full(20, 40_000.0)
        ])
        
        is_vcp, contraction_ratio, metrics = detect_volatility_contraction_pattern(
            highs, lows, closes, volumes, lookback=30
        )
        self.assertTrue(metrics["is_trend_aligned"])
        self.assertTrue(metrics["is_contracting"])
        self.assertTrue(metrics["is_volume_dry"])
        self.assertTrue(is_vcp)
        self.assertLess(contraction_ratio, 0.50)

    def test_vcp_negative_expanding_volatility(self):
        # Expanding volatility (widening swings) -> should reject VCP
        n = 40
        base_trend = np.linspace(100.0, 105.0, n)
        swings = np.concatenate([
            np.sin(np.linspace(0, 3.14 * 2, 20)) * 1.0,
            np.sin(np.linspace(0, 3.14 * 2, 20)) * 12.0
        ])
        closes = base_trend + swings
        highs = closes + 2.0
        lows = closes - 2.0
        volumes = np.full(40, 100_000.0)

        is_vcp, contraction_ratio, metrics = detect_volatility_contraction_pattern(
            highs, lows, closes, volumes
        )
        self.assertFalse(is_vcp)
        self.assertFalse(metrics["is_contracting"])

    def test_screener_filter_vcp_candidates(self):
        screener = QuantitativeScreener(RiskConfig())
        # Mock screened stocks
        from features.indicators import TechnicalMetrics
        from features.normalizer import FundamentalMetrics, MicrostructureMetrics
        
        dummy_tech = TechnicalMetrics(50, 0, 0, 0, 1, 0.01, 0, 0, 0, 0.5, 0.2, 1.0)
        dummy_fund = FundamentalMetrics(1e9, 20, 0.5, 0.1, 0.1, 1.0)
        dummy_micro = MicrostructureMetrics(0.001, 0, 0.99, 1.0)
        
        s1 = ScreenedStock("VCP_STOCK", 100.0, 1e9, 2e7, dummy_tech, dummy_fund, dummy_micro, True, is_vcp=True, vcp_ratio=0.3)
        s2 = ScreenedStock("NORMAL_STOCK", 50.0, 1e9, 2e7, dummy_tech, dummy_fund, dummy_micro, True, is_vcp=False, vcp_ratio=1.0)
        s3 = ScreenedStock("ILLIQUID_STOCK", 20.0, 1e7, 1e5, dummy_tech, dummy_fund, dummy_micro, False, is_vcp=True, vcp_ratio=0.2)
        
        vcp_list = screener.filter_vcp_candidates([s1, s2, s3])
        self.assertEqual(len(vcp_list), 1)
        self.assertEqual(vcp_list[0].ticker, "VCP_STOCK")

    def test_check_earnings_embargo(self):
        import datetime
        from unittest.mock import MagicMock
        screener = QuantitativeScreener(RiskConfig())

        # Mock ticker with earnings date 1 day from now
        mock_ticker = MagicMock()
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        mock_ticker.calendar = {"Earnings Date": [tomorrow]}

        embargoed, days, msg = screener.check_earnings_embargo(mock_ticker, days_embargo=2)
        self.assertTrue(embargoed)
        self.assertEqual(days, 1)
        self.assertIn("Binary earnings event scheduled in 1 days", msg)

        # Mock ticker with earnings date 10 days from now (safe)
        ten_days = datetime.date.today() + datetime.timedelta(days=10)
        mock_ticker.calendar = {"Earnings Date": [ten_days]}
        embargoed, days, msg = screener.check_earnings_embargo(mock_ticker, days_embargo=2)
        self.assertFalse(embargoed)
        self.assertIsNone(days)


if __name__ == "__main__":
    unittest.main()

