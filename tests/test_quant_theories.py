"""Unit tests for Institutional Quantitative Mathematical Feature Engine.

Verifies:
1. RMT Marčenko-Pastur spectral cutoff and eigenvalue denoising
2. Level-1 and Level-2 Rough Path Signature iterated integrals & Lévy area
3. Hurst exponent (distinguishing persistent trend vs mean-reversion)
4. Extreme Value Theory (EVT) Peaks-over-Threshold GPD fitting and VaR/ES
5. Transfer Entropy directional information causality
6. Minimum Spanning Tree (MST) distance and centrality
7. Thermodynamic Critical Slowing Down autocorrelation and variance
8. Topological Persistent Homology entropy proxy
9. Extreme Learning Machine (ELM) liquid reservoir projection
10. Unified feature extraction pipeline
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.quant_theories import QuantTheoryEngine, QuantTheoryFeatures


class TestQuantTheoryEngine(unittest.TestCase):

    def setUp(self):
        self.engine = QuantTheoryEngine(reservoir_dim=16, seed=42)

    def test_rmt_denoising(self):
        np.random.seed(42)
        T, N = 200, 10
        # Generate synthetic returns with 1 dominant market factor + pure Gaussian noise
        market_factor = np.random.normal(0, 0.02, (T, 1))
        noise = np.random.normal(0, 0.01, (T, N))
        returns = market_factor + noise

        cleaned_corr, top_evals, noise_ratio = self.engine.denoise_correlation_matrix_rmt(returns)

        self.assertEqual(cleaned_corr.shape, (N, N))
        # Diagonal must be 1.0
        np.testing.assert_allclose(np.diag(cleaned_corr), 1.0, atol=1e-5)
        # Dominant factor should create top eigenvalue >> 1
        self.assertTrue(top_evals[0] > 1.5)
        # Noise ratio should indicate bulk was cleaned
        self.assertTrue(0.0 <= noise_ratio <= 1.0)

    def test_path_signature_level2(self):
        np.random.seed(42)
        prices = np.linspace(100, 120, 30) + np.sin(np.linspace(0, 3.14, 30)) * 2.0
        volumes = np.linspace(1e5, 2e5, 30)

        sig = self.engine.compute_path_signature_level2(prices, volumes, window=20)
        self.assertEqual(sig.shape, (12,))
        self.assertFalse(np.isnan(sig).any())
        self.assertFalse(np.isinf(sig).any())

    def test_hurst_exponent_trending_vs_mean_reverting(self):
        np.random.seed(42)
        # Pure trending series (cumulative random walk with positive drift)
        trend = np.cumsum(np.random.normal(0.05, 0.02, 200)) + 100.0
        h_trend, lyap1 = self.engine.compute_hurst_exponent(trend)
        self.assertTrue(h_trend >= 0.5, f"Expected H >= 0.5 for trending, got {h_trend}")

        # Mean-reverting Ornstein-Uhlenbeck series
        ou = [100.0]
        for _ in range(199):
            ou.append(ou[-1] + 0.3 * (100.0 - ou[-1]) + np.random.normal(0, 1.0))
        h_ou, lyap2 = self.engine.compute_hurst_exponent(np.array(ou))
        self.assertTrue(h_ou <= 0.6, f"Expected H <= 0.6 for OU, got {h_ou}")

    def test_evt_tail_risk(self):
        np.random.seed(42)
        # Fat-tailed Student-t returns
        returns = np.random.standard_t(df=3, size=200) * 0.02

        xi, sigma, var_99, es_99 = self.engine.compute_evt_tail_risk(returns)
        self.assertTrue(sigma > 0.0)
        self.assertTrue(var_99 > 0.0)
        self.assertTrue(es_99 >= var_99, f"Expected ES ({es_99}) >= VaR ({var_99})")

    def test_transfer_entropy(self):
        np.random.seed(42)
        source = np.random.normal(0, 0.01, 100)
        # Target follows source with 1-lag delay
        target = np.roll(source, 1) + np.random.normal(0, 0.002, 100)

        te = self.engine.compute_transfer_entropy(source, target, bins=5)
        self.assertTrue(te > 0.0)

    def test_mst_centrality(self):
        np.random.seed(42)
        # 5 assets with known correlation
        corr = np.array([
            [1.0, 0.8, 0.8, 0.8, 0.8],
            [0.8, 1.0, 0.2, 0.2, 0.2],
            [0.8, 0.2, 1.0, 0.2, 0.2],
            [0.8, 0.2, 0.2, 1.0, 0.2],
            [0.8, 0.2, 0.2, 0.2, 1.0]
        ])
        # Asset 0 is the central hub connected to all other 4
        cent_0 = self.engine.compute_mst_centrality(corr, target_idx=0)
        cent_1 = self.engine.compute_mst_centrality(corr, target_idx=1)
        self.assertTrue(cent_0 > cent_1, f"Hub centrality ({cent_0}) must be > peripheral ({cent_1})")

    def test_critical_slowing_down(self):
        # Accelerating variance series
        s1 = np.random.normal(100, 1.0, 30)
        s2 = np.random.normal(100, 5.0, 30)
        series = np.concatenate([s1, s2])
        ac1, var_accel = self.engine.compute_critical_slowing_down(series, window=25)
        self.assertTrue(var_accel > 0.0)

    def test_topological_entropy_proxy(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 60)
        topo, w_dist = self.engine.compute_topological_entropy_proxy(returns, window=25)
        self.assertTrue(0.0 <= topo <= 1.0)
        self.assertTrue(w_dist >= 0.0)

    def test_elm_reservoir_projection(self):
        technicals = np.array([0.5, 0.1, 0.7, 0.2, 0.6, 0.04, 0.1, 0.2], dtype=np.float32)
        h = self.engine.project_elm_reservoir(technicals)
        self.assertEqual(h.shape, (16,))
        self.assertTrue((h >= -1.0).all() and (h <= 1.0).all())

    def test_extract_all_features_full_pipeline(self):
        prices = np.linspace(150, 180, 50) + np.random.normal(0, 1, 50)
        volumes = np.random.uniform(1e6, 5e6, 50)
        mkt_prices = np.linspace(400, 430, 50) + np.random.normal(0, 1, 50)

        feats = self.engine.extract_all_features(prices, volumes, market_index_prices=mkt_prices)
        self.assertIsInstance(feats, QuantTheoryFeatures)
        self.assertEqual(feats.signature_tensor.shape, (12,))
        self.assertEqual(feats.market_field_vector.shape, (10,))
        self.assertEqual(feats.chaos_evt_vector.shape, (10,))
        self.assertEqual(feats.elm_latent_state.shape, (16,))

    def test_rmt_ill_conditioned_stability(self):
        # Create an ill-conditioned singular return matrix (perfectly collinear assets)
        T, N = 100, 5
        base_series = np.random.normal(0, 0.01, (T, 1))
        # Identical columns yield rank 1 matrix with 4 zero eigenvalues
        collinear_returns = np.repeat(base_series, N, axis=1)

        cleaned_corr, top_evals, noise_ratio = self.engine.denoise_correlation_matrix_rmt(collinear_returns)

        # Check positive definiteness via Cholesky or eigenvalue floor
        evals = np.linalg.eigvalsh(cleaned_corr)
        self.assertTrue((evals > 0).all(), "Cleaned matrix must be strictly positive definite")
        self.assertFalse(np.isnan(cleaned_corr).any())
        self.assertFalse(np.isinf(cleaned_corr).any())
        # Condition number must be well-bounded
        cond_num = np.linalg.cond(cleaned_corr)
        self.assertLess(cond_num, 1e4)

    def test_cvd_divergence_calculation(self):
        # Test Cumulative Volume Delta absorption calculation
        highs = np.array([100, 102, 101, 103, 105, 104, 106, 107, 108, 110], dtype=float)
        lows = np.array([98, 99, 99, 100, 102, 101, 103, 104, 105, 107], dtype=float)
        closes = np.array([99, 101, 100, 102, 103, 102, 105, 106, 107, 109], dtype=float)
        volumes = np.array([1000, 1200, 1100, 1500, 1800, 1400, 2000, 2200, 2100, 2500], dtype=float)

        cvd, div = QuantTheoryEngine.compute_cumulative_volume_delta(highs, lows, closes, volumes)
        self.assertIsInstance(cvd, float)
        self.assertIsInstance(div, float)
        self.assertTrue(-1.0 <= cvd <= 1.0)
        self.assertIn(div, (-1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
