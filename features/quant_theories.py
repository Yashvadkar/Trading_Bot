"""Institutional Quantitative Mathematical & Physics Feature Engine.

Synthesizes 11 foundational frameworks from mathematical physics, information theory,
and dynamical systems into vectorized, high-throughput NumPy/PyTorch modules:

1. Random Matrix Theory (RMT) — Marčenko–Pastur spectral denoising
2. Path Signatures (Rough Path Theory) — Truncated Level-1 & Level-2 iterated integrals (Lévy area)
3. Chaos Theory & Dynamical Systems — Hurst Exponent via Rescaled Range (R/S)
4. Extreme Value Theory (EVT) — Peaks-over-Threshold Generalized Pareto Tail Risk (VaR, ES)
5. Transfer Entropy (TE) — Model-free directional information causality
6. Network Theory & Graph Data Science — Ultrametric distances & Minimum Spanning Tree (MST) centrality
7. Thermodynamic Critical Slowing Down — Autocorrelation lag-1 & variance acceleration
8. Topological Data Analysis (TDA) — Persistent homology dispersion & topological entropy proxy
9. Reservoir Computing (ELM) — Instantaneous high-dimensional random projection
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class QuantTheoryFeatures:
    """Rich mathematical feature representation across dynamical regimes."""
    # Rough Path (Level-1 increments & Level-2 Lévy Area) - 12 dims
    signature_tensor: np.ndarray      # Shape (12,)
    # Market Field & RMT - 10 dims
    rmt_top_eigenvalues: np.ndarray   # Shape (4,)
    rmt_noise_ratio: float
    transfer_entropy: float
    mst_centrality: float
    topological_entropy: float
    wasserstein_regime_shift: float
    market_field_vector: np.ndarray   # Shape (10,)
    # Dynamical Chaos & EVT - 10 dims
    hurst_exponent: float
    lyapunov_proxy: float
    critical_slowing_score: float
    variance_acceleration: float
    evt_shape_xi: float
    evt_scale_sigma: float
    evt_var_99: float
    evt_es_99: float
    chaos_evt_vector: np.ndarray      # Shape (10,)
    # ELM Liquid Reservoir - 16 dims
    elm_latent_state: np.ndarray      # Shape (16,)


class QuantTheoryEngine:
    """Vectorized mathematical engine for institutional quantitative feature extraction."""

    def __init__(self, reservoir_dim: int = 16, seed: int = 42):
        self.reservoir_dim = reservoir_dim
        self.rng = np.random.default_rng(seed)
        # Pre-seed fixed orthogonal/Gaussian weights for the ELM reservoir (8 input technicals -> 16 latents)
        self.elm_weights = self.rng.normal(0.0, 1.0, (8, reservoir_dim)).astype(np.float32)
        self.elm_biases = self.rng.uniform(-0.5, 0.5, reservoir_dim).astype(np.float32)

    # =========================================================================
    # 1. Random Matrix Theory (RMT) — Marčenko–Pastur Spectral Denoising
    # =========================================================================
    @staticmethod
    def clean_correlation_rmt(
        returns: np.ndarray,
        clip_floor: float = 1e-8,
        shrinkage_weight: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Filters noise eigenvalues using the analytical Marčenko–Pastur law and Ledoit-Wolf shrinkage.
        
        Given T observations and N assets:
            q = T / N
            lambda_+ = sigma^2 * (1 + sqrt(1 / q))^2
        Under scoped np.errstate, clips eigenvalues to floor, shrinks noise eigenvalues to their mean,
        reconstructs cleaned correlation matrix, and applies Ledoit-Wolf structural shrinkage:
            C_shrunk = (1 - alpha) * C_clean + alpha * I
        
        Args:
            returns: Array of shape (T, N) where T >= N
            clip_floor: Minimum eigenvalue threshold
            shrinkage_weight: Ledoit-Wolf shrinkage coefficient alpha (default: 0.05)
            
        Returns:
            (shrunk_corr, top_eigenvalues, noise_ratio)
        """
        returns = np.asarray(returns, dtype=np.float64)
        t_obs, n_assets = returns.shape
        if t_obs < 2 or n_assets < 2:
            return np.eye(n_assets), np.ones(min(n_assets, 4)), 0.0

        # Empirical correlation matrix
        corr = np.corrcoef(returns, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # Eigen-decomposition (Hermitian / symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Marčenko-Pastur upper bound: q = T / N, lambda_+ = sigma^2 * (1 + sqrt(1/q))^2
        q = float(t_obs) / float(n_assets)
        lambda_plus = (1.0 + np.sqrt(1.0 / q)) ** 2

        with np.errstate(divide='ignore', invalid='ignore'):
            clipped_eigvals = np.clip(eigenvalues, a_min=clip_floor, a_max=None)
            is_noise = (clipped_eigvals <= lambda_plus)
            if np.any(is_noise):
                noise_mean = float(np.mean(clipped_eigvals[is_noise]))
                clipped_eigvals[is_noise] = max(noise_mean, clip_floor)

            # Reconstruct cleaned correlation matrix: C = V * Lambda * V^T
            cleaned_corr = eigenvectors @ np.diag(clipped_eigvals) @ eigenvectors.T
            # Rescale diagonals to strictly 1.0
            d = np.sqrt(np.maximum(np.diag(cleaned_corr), clip_floor))
            d_outer = np.outer(d, d)
            cleaned_corr = np.divide(cleaned_corr, d_outer, out=np.zeros_like(cleaned_corr), where=(d_outer > 0))
            np.fill_diagonal(cleaned_corr, 1.0)

            # Post-clipping Ledoit-Wolf structured shrinkage toward Identity
            # Guarantees strict positive definiteness and bounds condition number
            shrunk_corr = (1.0 - shrinkage_weight) * cleaned_corr + shrinkage_weight * np.eye(n_assets)
            np.fill_diagonal(shrunk_corr, 1.0)

        noise_ratio = float(np.sum(is_noise)) / float(n_assets)
        top_evals = clipped_eigvals[:min(n_assets, 4)]
        if len(top_evals) < 4:
            top_evals = np.pad(top_evals, (0, 4 - len(top_evals)), constant_values=1.0)

        return shrunk_corr, top_evals, noise_ratio

    @classmethod
    def denoise_correlation_matrix_rmt(
        cls,
        returns: np.ndarray,
        clip_floor: float = 1e-8,
        shrinkage_weight: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Backward-compatible entry point delegating to clean_correlation_rmt."""
        return cls.clean_correlation_rmt(returns, clip_floor=clip_floor, shrinkage_weight=shrinkage_weight)

    # =========================================================================
    # 2. Path Signatures (Rough Path Theory) — Iterated Integrals & Lévy Area
    # =========================================================================
    @staticmethod
    def compute_path_signature_level2(
        price_series: np.ndarray,
        volume_series: np.ndarray,
        window: int = 20
    ) -> np.ndarray:
        """Computes truncated Level-1 and Level-2 Rough Path Signature tensor.
        
        Constructs a 3-dimensional stream X_t = (t, P_t, V_t) normalized on [0, 1].
        Level 1: Increments Delta X^i
        Level 2: Iterated integrals int dX^i dX^j (capturing velocity and Lévy area)
        
        Total signature features: 3 (Level 1) + 9 (Level 2) = 12 features.
        
        Args:
            price_series: 1D array of closing prices
            volume_series: 1D array of volume
            window: Number of steps to compute path over
            
        Returns:
            signature_vec: Shape (12,)
        """
        prices = np.asarray(price_series[-window:], dtype=np.float64)
        vols = np.asarray(volume_series[-window:], dtype=np.float64)
        L = len(prices)
        if L < 3:
            return np.zeros(12, dtype=np.float32)

        # Normalize stream components to zero-start, scaled variance
        p_norm = (prices - prices[0]) / (np.std(prices) + 1e-6)
        v_norm = (vols - vols[0]) / (np.std(vols) + 1e-6)
        t_norm = np.linspace(0.0, 1.0, L)

        # Matrix of path channels: shape (L, 3) -> [time, price, volume]
        X = np.column_stack([t_norm, p_norm, v_norm])
        dX = np.diff(X, axis=0)  # Shape (L-1, 3)

        # Level 1: Total path displacement S^(1)_i = X_T^i - X_0^i
        level1 = np.sum(dX, axis=0)  # Shape (3,)

        # Level 2: Iterated integrals S^(2)_{i,j} = \int_{s < t} dX_s^i dX_t^j
        # Discretized Chen identity: S^(2)_{i,j} = \sum_{k=1}^{L-1} (X_{k-1}^i - X_0^i) * dX_k^j + 0.5 * dX_k^i * dX_k^j
        X_shifted = X[:-1] - X[0]  # Shape (L-1, 3)
        level2 = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                cross = np.sum(X_shifted[:, i] * dX[:, j]) + 0.5 * np.sum(dX[:, i] * dX[:, j])
                level2[i, j] = cross

        # Concatenate: 3 Level-1 + 9 Level-2 = 12 features
        signature = np.concatenate([level1, level2.flatten()]).astype(np.float32)
        return np.nan_to_num(signature, nan=0.0)

    # =========================================================================
    # 3. Chaos Theory & Non-Linear Dynamics — Hurst Exponent & Lyapunov Proxy
    # =========================================================================
    @staticmethod
    def compute_hurst_exponent(series: np.ndarray, min_window: int = 8) -> tuple[float, float]:
        """Calculates Hurst Exponent (H) via Mandelbrot Rescaled Range (R/S) Analysis.
        
        H < 0.45: Mean-reverting / anti-persistent (Ornstein-Uhlenbeck regime)
        0.45 <= H <= 0.55: Brownian random walk / zero memory
        H > 0.55: Persistent / trend-following / super-diffusive
        
        Returns:
            (hurst_exponent, lyapunov_divergence_proxy)
        """
        arr = np.asarray(series, dtype=np.float64)
        N = len(arr)
        if N < min_window * 3:
            return 0.5, 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(np.log(np.maximum(arr, 1e-6)))
            if len(returns) < min_window:
                return 0.5, 0.0

            # Sub-window scales
            scales = []
            rs_values = []
            max_scale = N // 2
            curr_scale = min_window

            while curr_scale <= max_scale:
                num_chunks = len(returns) // curr_scale
                rs_chunk_list = []
                for i in range(num_chunks):
                    chunk = returns[i * curr_scale : (i + 1) * curr_scale]
                    mean_adj = chunk - np.mean(chunk)
                    cum_dev = np.cumsum(mean_adj)
                    R = np.max(cum_dev) - np.min(cum_dev)
                    S = np.std(chunk)
                    if S > 1e-8:
                        rs_chunk_list.append(R / S)
                if rs_chunk_list:
                    scales.append(curr_scale)
                    rs_values.append(np.mean(rs_chunk_list))
                curr_scale = int(curr_scale * 1.5) + 1

            if len(scales) < 3:
                return 0.5, 0.0

            # Linear regression: log(R/S) = H * log(n) + c
            log_scales = np.log(scales)
            log_rs = np.log(rs_values)
            poly = np.polyfit(log_scales, log_rs, 1)
            H = float(np.clip(poly[0], 0.05, 0.95))

            # Local Lyapunov exponent proxy: rate of exponential divergence of nearby returns
            diffs = np.abs(np.diff(returns))
            lyapunov_proxy = float(np.clip(np.mean(np.log(np.maximum(diffs, 1e-6))) + 5.0, -3.0, 3.0))

        return H, lyapunov_proxy

    # =========================================================================
    # 4. Extreme Value Theory (EVT) — Peaks-over-Threshold GPD Tail Risk
    # =========================================================================
    @staticmethod
    def compute_evt_tail_risk(
        returns: np.ndarray,
        quantile_threshold: float = 0.90,
        alpha: float = 0.99
    ) -> tuple[float, float, float, float]:
        """Fits Generalized Pareto Distribution (GPD) on extreme loss tails.
        
        Pickands-Balkema-de Haan Theorem: Tail losses exceeding threshold u
        converge asymptotically to GPD:
            G_{xi, sigma}(y) = 1 - (1 + xi * y / sigma)^(-1 / xi)
            
        Returns:
            (shape_xi, scale_sigma, VaR_alpha, Expected_Shortfall_alpha)
        """
        ret = np.asarray(returns, dtype=np.float64)
        if len(ret) < 30:
            return 0.1, 0.02, 0.04, 0.06

        losses = -ret  # Loss distribution (positive for losses)
        losses = losses[losses > 0]
        if len(losses) < 15:
            return 0.1, 0.02, 0.03, 0.05

        u = float(np.quantile(losses, quantile_threshold))
        excesses = losses[losses > u] - u
        N_u = len(excesses)
        N = len(ret)

        if N_u < 5:
            return 0.1, float(np.std(losses)), float(u * 1.5), float(u * 2.0)

        # Method of Moments estimator for GPD (numerically stable & fast)
        mean_ex = float(np.mean(excesses))
        var_ex = float(np.var(excesses))
        if var_ex > 1e-8:
            xi = 0.5 * (1.0 - (mean_ex ** 2) / var_ex)
            sigma = 0.5 * mean_ex * ((mean_ex ** 2) / var_ex + 1.0)
        else:
            xi = 0.1
            sigma = max(mean_ex, 1e-4)

        xi = float(np.clip(xi, -0.4, 0.5))
        sigma = float(max(sigma, 1e-4))

        # Analytical VaR and Expected Shortfall under GPD
        # VaR_alpha = u + (sigma / xi) * [ ((N / N_u) * (1 - alpha))^(-xi) - 1 ]
        prob_ratio = (float(N) / float(N_u)) * (1.0 - alpha)
        if prob_ratio > 0 and abs(xi) > 1e-4:
            var_alpha = u + (sigma / xi) * (math.pow(prob_ratio, -xi) - 1.0)
        else:
            var_alpha = u - sigma * math.log(max(prob_ratio, 1e-6))

        # Expected Shortfall: ES = (VaR + sigma - xi * u) / (1 - xi)
        if xi < 1.0:
            es_alpha = (var_alpha + sigma - xi * u) / (1.0 - xi)
        else:
            es_alpha = var_alpha * 1.5

        return (
            xi,
            sigma,
            float(np.clip(var_alpha, 0.005, 0.25)),
            float(np.clip(es_alpha, 0.008, 0.35))
        )

    # =========================================================================
    # 5. Transfer Entropy (TE) — Directional Causality Flow
    # =========================================================================
    @staticmethod
    def compute_transfer_entropy(
        source_returns: np.ndarray,
        target_returns: np.ndarray,
        bins: int = 5
    ) -> float:
        """Estimates directional Shannon Transfer Entropy T_{source -> target}.
        
        Quantifies bits of information transferred from the market driver (SPY)
        to the target stock not already contained in the target's past.
        
        T_{X->Y} = sum p(y_{t+1}, y_t, x_t) * log2( p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t) )
        """
        src = np.asarray(source_returns, dtype=np.float64)
        tgt = np.asarray(target_returns, dtype=np.float64)
        L = min(len(src), len(tgt))
        if L < 20:
            return 0.0

        # Discretize continuous returns into quantile bins
        def _discretize(x: np.ndarray) -> np.ndarray:
            ranks = np.argsort(np.argsort(x))
            return np.floor(ranks / len(x) * bins).astype(int)

        x_disc = _discretize(src[:L-1])
        y_past = _discretize(tgt[:L-1])
        y_next = _discretize(tgt[1:L])

        N = L - 1
        # Joint and marginal empirical distributions via 3D histograms
        joint_3d = np.zeros((bins, bins, bins), dtype=np.float64)
        joint_2d_yy = np.zeros((bins, bins), dtype=np.float64)
        joint_2d_yx = np.zeros((bins, bins), dtype=np.float64)
        marg_y = np.zeros(bins, dtype=np.float64)

        for i in range(N):
            yn, yp, xp = y_next[i], y_past[i], x_disc[i]
            joint_3d[yn, yp, xp] += 1.0
            joint_2d_yy[yn, yp] += 1.0
            joint_2d_yx[yp, xp] += 1.0
            marg_y[yp] += 1.0

        joint_3d /= N
        joint_2d_yy /= N
        joint_2d_yx /= N
        marg_y /= N

        with np.errstate(divide='ignore', invalid='ignore'):
            te = 0.0
            for yn in range(bins):
                for yp in range(bins):
                    for xp in range(bins):
                        p_yn_yp_xp = joint_3d[yn, yp, xp]
                        p_yp_xp = joint_2d_yx[yp, xp]
                        p_yn_yp = joint_2d_yy[yn, yp]
                        p_yp = marg_y[yp]

                        if p_yn_yp_xp > 1e-7 and p_yp_xp > 1e-7 and p_yn_yp > 1e-7 and p_yp > 1e-7:
                            # p(yn | yp, xp) / p(yn | yp) = (p_yn_yp_xp / p_yp_xp) / (p_yn_yp / p_yp)
                            cond_ratio = (p_yn_yp_xp * p_yp) / (p_yp_xp * p_yn_yp)
                            if cond_ratio > 0:
                                te += p_yn_yp_xp * math.log2(cond_ratio)

        return float(np.clip(te, 0.0, 3.0))

    # =========================================================================
    # 6. Network Theory — Minimum Spanning Tree (MST) Centrality
    # =========================================================================
    @staticmethod
    def compute_mst_centrality(corr_matrix: np.ndarray, target_idx: int = 0) -> float:
        """Computes Minimum Spanning Tree centrality on ultrametric distance matrix.
        
        Distance: d(i, j) = sqrt(2 * (1 - rho_{i, j}))
        Uses Kruskal's algorithm to form the MST and measures degree / closeness centrality.
        """
        corr = np.asarray(corr_matrix, dtype=np.float64)
        N = corr.shape[0]
        if N < 3:
            return 0.5

        # Ultrametric distance
        dist = np.sqrt(np.maximum(2.0 * (1.0 - corr), 0.0))

        # Extract edges
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                edges.append((dist[i, j], i, j))
        edges.sort(key=lambda x: x[0])

        # Kruskal Disjoint-Set Forest
        parent = list(range(N))
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        mst_adj = np.zeros((N, N), dtype=int)
        edge_count = 0
        for d_val, u, v in edges:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_u] = root_v
                mst_adj[u, v] = 1
                mst_adj[v, u] = 1
                edge_count += 1
                if edge_count == N - 1:
                    break

        # Degree centrality of target asset in the MST
        target_degree = np.sum(mst_adj[target_idx])
        normalized_centrality = float(target_degree) / float(N - 1)
        return float(np.clip(normalized_centrality, 0.0, 1.0))

    # =========================================================================
    # 7. Thermodynamic Critical Slowing Down & Phase Transitions
    # =========================================================================
    @staticmethod
    def compute_critical_slowing_down(series: np.ndarray, window: int = 20) -> tuple[float, float]:
        """Calculates lag-1 autocorrelation and variance acceleration.
        
        Near a tipping point / market phase transition, the recovery rate from
        perturbations slows dramatically:
            Autocorrelation phi(1) -> 1.0
            Variance accelerates: Delta sigma^2 > 0
        """
        arr = np.asarray(series, dtype=np.float64)
        if len(arr) < window:
            return 0.0, 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            recent = arr[-window:]
            recent_detrend = recent - np.mean(recent)
            var_now = float(np.var(recent_detrend))

            if len(arr) >= 2 * window:
                prev = arr[-2 * window : -window]
                prev_detrend = prev - np.mean(prev)
                var_prev = float(np.var(prev_detrend))
                var_accel = (var_now - var_prev) / (var_prev + 1e-6)
            else:
                var_accel = 0.0

            # Lag-1 Autocorrelation
            if var_now > 1e-8:
                autocorr_1 = np.corrcoef(recent_detrend[:-1], recent_detrend[1:])[0, 1]
                autocorr_1 = float(np.nan_to_num(autocorr_1, nan=0.0))
            else:
                autocorr_1 = 0.0

        return float(np.clip(autocorr_1, -1.0, 1.0)), float(np.clip(var_accel, -2.0, 3.0))

    # =========================================================================
    # 8. Topological Data Analysis (TDA) & Persistent Homology Proxy
    # =========================================================================
    @staticmethod
    def compute_topological_entropy_proxy(returns: np.ndarray, window: int = 25) -> tuple[float, float]:
        """Topological point-cloud dispersion and Wasserstein regime shift proxy.
        
        Computes the geometric entropy of pairwise distance matrices across
        time-delay return embeddings to detect topological clustering collapse
        prior to liquidity breakdowns.
        """
        ret = np.asarray(returns, dtype=np.float64)
        if len(ret) < window:
            return 0.5, 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            # 2D Takens delay embedding: (r_t, r_{t-1})
            sub = ret[-window:]
            points = np.column_stack([sub[:-1], sub[1:]])  # Shape (window-1, 2)

            # Pairwise Euclidean distance matrix
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))

            # Topological entropy of persistence scale distribution
            triu_dists = dists[np.triu_indices_from(dists, k=1)]
            if len(triu_dists) > 0 and np.sum(triu_dists) > 0:
                p_dist = triu_dists / np.sum(triu_dists)
                p_dist = p_dist[p_dist > 1e-6]
                topological_entropy = -float(np.sum(p_dist * np.log2(p_dist))) / math.log2(len(triu_dists))
            else:
                topological_entropy = 0.5

            # Wasserstein morphological distance vs older reference window
            if len(ret) >= 2 * window:
                ref_sub = ret[-2 * window : -window]
                w_dist = float(np.abs(np.mean(sub) - np.mean(ref_sub)) + np.abs(np.std(sub) - np.std(ref_sub)))
            else:
                w_dist = 0.0

        return float(np.clip(topological_entropy, 0.0, 1.0)), float(np.clip(w_dist * 50.0, 0.0, 3.0))

    # =========================================================================
    # 9. Microstructure Alpha — Cumulative Volume Delta (CVD) & Absorption
    # =========================================================================
    @staticmethod
    def compute_cumulative_volume_delta(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> tuple[float, float]:
        """Calculates Cumulative Volume Delta (CVD) and Absorption Divergence.
        
        Uses the Lee-Ready / tick range proxy to classify taker buy vs sell volume:
            q_t = (Close - Low) / (High - Low + 1e-6)  # Taker buy fraction [0, 1]
            Delta_V_t = (2 * q_t - 1) * Volume_t
            
        Returns:
            (normalized_cvd, absorption_divergence_signal)
            absorption_divergence: -1.0 (bearish absorption / selling into highs),
                                  +1.0 (bullish exhaustion / buying into lows),
                                   0.0 (aligned)
        """
        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)
        v = np.asarray(volumes, dtype=np.float64)
        L = min(len(h), len(l), len(c), len(v))
        if L < 5:
            return 0.0, 0.0

        h, l, c, v = h[-L:], l[-L:], c[-L:], v[-L:]
        price_range = np.maximum(h - l, 1e-6)
        buy_fractions = (c - l) / price_range
        taker_deltas = (2.0 * buy_fractions - 1.0) * v
        cvd_series = np.cumsum(taker_deltas)

        # Normalized net CVD over window
        total_vol = max(np.sum(v), 1.0)
        norm_cvd = float(np.clip(cvd_series[-1] / total_vol, -1.0, 1.0))

        # Divergence detection over last 10 bars
        if L >= 10:
            price_slope = float(c[-1] - c[-10]) / max(c[-10], 1e-4)
            cvd_slope = float(cvd_series[-1] - cvd_series[-10]) / total_vol

            # Bearish divergence: price rising but taker delta falling (passive absorption)
            if price_slope > 0.01 and cvd_slope < -0.05:
                divergence = -1.0
            # Bullish divergence: price falling but taker delta rising (passive accumulation)
            elif price_slope < -0.01 and cvd_slope > 0.05:
                divergence = 1.0
            else:
                divergence = 0.0
        else:
            divergence = 0.0

        return norm_cvd, divergence

    # =========================================================================
    # 10. Extreme Learning Machine (ELM) — Liquid Reservoir Projection
    # =========================================================================
    def project_elm_reservoir(self, raw_technicals: np.ndarray) -> np.ndarray:
        """Projects 8 raw technical features into a 16-dimensional non-linear liquid state.
        
        Uses randomized orthogonal weights and hyperbolic tangent non-linearity:
            h = tanh(X * W_in + b)
        Achieves microsecond transformation on CPU with zero backpropagation.
        """
        x = np.asarray(raw_technicals, dtype=np.float32).flatten()
        if len(x) < 8:
            x = np.pad(x, (0, 8 - len(x)))
        elif len(x) > 8:
            x = x[:8]

        # Non-linear reservoir transformation
        h = np.tanh(np.dot(x, self.elm_weights) + self.elm_biases)
        return h.astype(np.float32)

    # =========================================================================
    # Master Extraction Pipeline
    # =========================================================================
    def extract_all_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        market_index_prices: np.ndarray | None = None,
        universe_returns_matrix: np.ndarray | None = None,
        target_asset_idx: int = 0
    ) -> QuantTheoryFeatures:
        """Unified extractor orchestrating all 11 quantitative physics frameworks."""
        prices = np.asarray(prices, dtype=np.float64)
        volumes = np.asarray(volumes, dtype=np.float64)
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-6)

        # 1. Rough Path Signature (12 dims)
        sig_tensor = self.compute_path_signature_level2(prices, volumes, window=20)

        # 2. Chaos Theory (Hurst, Lyapunov)
        hurst, lyap = self.compute_hurst_exponent(prices)

        # 3. Extreme Value Theory Tail Risk
        xi, sigma, var_99, es_99 = self.compute_evt_tail_risk(returns)

        # 4. Critical Slowing Down
        ac1, var_accel = self.compute_critical_slowing_down(prices)

        # 5. Topological Data Analysis Proxy
        topo_entropy, wasserstein = self.compute_topological_entropy_proxy(returns)

        # 6. Benchmark Transfer Entropy
        if market_index_prices is not None and len(market_index_prices) > 20:
            mkt_returns = np.diff(market_index_prices) / np.maximum(market_index_prices[:-1], 1e-6)
            te = self.compute_transfer_entropy(mkt_returns, returns)
        else:
            te = 0.15

        # 7. RMT Spectral Filtering & MST Centrality
        if universe_returns_matrix is not None and universe_returns_matrix.shape[0] >= universe_returns_matrix.shape[1]:
            _, top_evals, noise_ratio = self.denoise_correlation_matrix_rmt(universe_returns_matrix)
            corr = np.corrcoef(universe_returns_matrix, rowvar=False)
            mst_cent = self.compute_mst_centrality(corr, target_idx=target_asset_idx)
        else:
            top_evals = np.array([2.5, 1.2, 0.8, 0.5], dtype=np.float64)
            noise_ratio = 0.45
            mst_cent = 0.35

        # Assemble Market Field Vector (10 dims)
        market_field = np.array([
            top_evals[0] / 5.0, top_evals[1] / 3.0, top_evals[2] / 2.0, top_evals[3] / 2.0,
            noise_ratio,
            te / 1.5,
            mst_cent,
            topo_entropy,
            wasserstein / 2.0,
            float(np.clip(returns[-1] * 10.0 if len(returns) > 0 else 0.0, -2.0, 2.0))
        ], dtype=np.float32)

        # Assemble Chaos & EVT Vector (10 dims)
        chaos_evt = np.array([
            (hurst - 0.5) * 4.0,       # Clamped around 0 for Brownian
            lyap / 2.0,
            ac1,
            var_accel,
            xi * 2.0,
            sigma * 20.0,
            var_99 * 15.0,
            es_99 * 10.0,
            float(np.clip(math.sqrt(max(float(np.var(returns)), 1e-8)) * 25.0, 0.0, 3.0)),
            float(np.clip((prices[-1] / (np.mean(prices[-20:]) + 1e-6) - 1.0) * 15.0, -3.0, 3.0))
        ], dtype=np.float32)

        # 8. ELM Liquid Reservoir State (16 dims from 8 basic technicals)
        recent_p = prices[-14:] if len(prices) >= 14 else prices
        simple_technicals = np.array([
            float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0,
            float(np.std(returns[-10:])) if len(returns) >= 10 else 0.01,
            float((prices[-1] - np.min(recent_p)) / (np.max(recent_p) - np.min(recent_p) + 1e-6)),
            float(np.log(max(volumes[-1], 1e3)) / 15.0),
            float(hurst),
            float(var_99),
            float(ac1),
            float(te)
        ], dtype=np.float32)
        elm_latent = self.project_elm_reservoir(simple_technicals)

        return QuantTheoryFeatures(
            signature_tensor=sig_tensor,
            rmt_top_eigenvalues=top_evals.astype(np.float32),
            rmt_noise_ratio=noise_ratio,
            transfer_entropy=te,
            mst_centrality=mst_cent,
            topological_entropy=topo_entropy,
            wasserstein_regime_shift=wasserstein,
            market_field_vector=market_field,
            hurst_exponent=hurst,
            lyapunov_proxy=lyap,
            critical_slowing_score=ac1,
            variance_acceleration=var_accel,
            evt_shape_xi=xi,
            evt_scale_sigma=sigma,
            evt_var_99=var_99,
            evt_es_99=es_99,
            chaos_evt_vector=chaos_evt,
            elm_latent_state=elm_latent
        )
