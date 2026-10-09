import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.governor import (
    calculate_conformal_volatility,
    calculate_fractional_kelly_capital,
    allocate_portfolio_with_cash_preservation
)


class TestConformalKelly(unittest.TestCase):

    def test_conformal_volatility_calculation(self):
        # Known residual distribution
        # e.g. Absolute residuals uniformly spread up to 0.04
        # 80th percentile should be ~0.032, scaled by z_0.90 (1.2816) -> ~0.025
        np.random.seed(42)
        residuals = np.random.normal(loc=0.0, scale=0.02, size=100)
        sigma_hat = calculate_conformal_volatility(residuals, alpha=0.20, z_score=1.2816)

        # Should be closely aligned with underlying distribution scale (~0.02)
        self.assertGreater(sigma_hat, 0.015)
        self.assertLess(sigma_hat, 0.035)

    def test_conformal_kelly_scales_with_uncertainty(self):
        equity = 100_000.0
        # Modest return drift (mu = ~0.24% per bar)
        returns = np.array([0.002, 0.003, 0.0025, 0.002, 0.0025])

        # Narrow residuals (high predictive certainty)
        narrow_residuals = np.array([0.002, -0.003, 0.001, -0.002, 0.003])
        cap_narrow = calculate_fractional_kelly_capital(
            returns=returns,
            portfolio_equity=equity,
            kappa=0.15,
            conformal_residuals=narrow_residuals
        )

        # Wide residuals (low predictive certainty / high regime noise)
        wide_residuals = np.array([0.10, -0.12, 0.15, -0.11, 0.14])
        cap_wide = calculate_fractional_kelly_capital(
            returns=returns,
            portfolio_equity=equity,
            kappa=0.15,
            conformal_residuals=wide_residuals
        )

        # Narrow confidence must allocate significantly more capital than wide confidence
        self.assertGreater(cap_narrow, cap_wide)
        # Wide residuals should severely compress allocation
        self.assertLess(cap_wide, cap_narrow * 0.2)

    def test_cash_preservation_avoids_gross_cap_washout(self):
        target_fractions = {"AAPL": 0.20, "NVDA": 0.25, "MSFT": 0.15}
        # Normal regime (uncertainty_scale = 1.0)
        normal_alloc = allocate_portfolio_with_cash_preservation(
            target_fractions, max_leverage=1.5, uncertainty_scale=1.0
        )
        # Total requested = 0.60 <= 1.5, so positions remain intact; 0.40 is in CASH
        self.assertAlmostEqual(sum(normal_alloc.values()), 0.60)
        self.assertAlmostEqual(normal_alloc["AAPL"], 0.20)

        # High systemic market uncertainty (uncertainty_scale = 2.0)
        # Effective gross cap drops from 1.5 to 1.5 / 2.0 = 0.75
        # If requested total was 1.0 > 0.75, it scales down and preserves cash
        stressed_targets = {"AAPL": 0.40, "NVDA": 0.40, "MSFT": 0.40}  # Total 1.20
        stressed_alloc = allocate_portfolio_with_cash_preservation(
            stressed_targets, max_leverage=1.5, uncertainty_scale=2.0
        )
        self.assertAlmostEqual(sum(stressed_alloc.values()), 0.75)
        # Verify positions scaled down proportionally and CASH increased
        self.assertLess(stressed_alloc["AAPL"], 0.40)


if __name__ == "__main__":
    unittest.main()
