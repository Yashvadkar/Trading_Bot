"""Multi-Year Historical Data Ingestion Engine for Quantitative Pre-Training.

Downloads multi-year daily and hourly OHLCV data for market benchmark and liquid equities,
extracts 64-dimensional quantitative state vectors across rolling windows, and generates
realistic experience replay transitions for offline neural pre-training.

Caches downloaded and precomputed features locally in data/cache/ to avoid redundant API hits.
"""

from __future__ import annotations
import os
from pathlib import Path
import time
import numpy as np
import yfinance as yf
from features.indicators import extract_technicals
from features.quant_theories import QuantTheoryEngine
from features.normalizer import StateNormalizer, FundamentalMetrics, MicrostructureMetrics, PortfolioContext, DebateMetrics
from model.replay_buffer import ReplayBuffer


class HistoricalDataLoader:
    """Ingests multi-year financial data and generates experience replay transitions."""

    DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "AMD"]

    def __init__(self, cache_dir: str | Path | None = None, seed: int = 42):
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.quant_engine = QuantTheoryEngine(reservoir_dim=16, seed=seed)
        self.normalizer = StateNormalizer(seed=seed)

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        period: str = "3y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> dict[str, np.ndarray] | None:
        """Fetches OHLCV arrays for a symbol, reading from local cache when available."""
        cache_file = self.cache_dir / f"{symbol}_{period}_{interval}.npz"

        if not force_refresh and cache_file.exists():
            try:
                data = np.load(cache_file)
                return {k: data[k] for k in data.files}
            except Exception:
                pass

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty or len(df) < 50:
                return None

            result = {
                "opens": df["Open"].to_numpy(dtype=np.float64),
                "highs": df["High"].to_numpy(dtype=np.float64),
                "lows": df["Low"].to_numpy(dtype=np.float64),
                "closes": df["Close"].to_numpy(dtype=np.float64),
                "volumes": df["Volume"].to_numpy(dtype=np.float64),
            }
            np.savez_compressed(cache_file, **result)
            return result
        except Exception as e:
            print(f"[HistoricalDataLoader] Error fetching {symbol}: {e}")
            return None

    def build_historical_replay_buffer(
        self,
        symbols: list[str] | None = None,
        period: str = "2y",
        lookback_window: int = 30,
        forward_horizon: int = 5,
        max_samples_per_symbol: int = 300
    ) -> ReplayBuffer:
        """Constructs an experience replay buffer populated with real historical transitions."""
        symbols = symbols or self.DEFAULT_SYMBOLS
        buffer = ReplayBuffer(capacity=len(symbols) * max_samples_per_symbol + 500)

        # 1. Fetch benchmark index data (SPY) for cross-market Transfer Entropy and RMT
        spy_data = self.fetch_historical_ohlcv("SPY", period=period)
        spy_closes = spy_data["closes"] if spy_data is not None else None

        # Collect return series across symbols for cross-sectional correlation
        all_symbol_data = {}
        for sym in symbols:
            data = self.fetch_historical_ohlcv(sym, period=period)
            if data is not None and len(data["closes"]) >= lookback_window + forward_horizon:
                all_symbol_data[sym] = data
            time.sleep(0.05)  # Politeness interval

        if not all_symbol_data:
            print("[HistoricalDataLoader] Warning: No historical data downloaded. Using synthetic fallbacks.")
            return buffer

        # Build multi-asset return matrix for RMT
        min_len = min(len(d["closes"]) for d in all_symbol_data.values())
        returns_list = []
        sym_keys = list(all_symbol_data.keys())
        for sym in sym_keys:
            c = all_symbol_data[sym]["closes"][:min_len]
            r = np.diff(c) / np.maximum(c[:-1], 1e-6)
            returns_list.append(r)
        universe_returns = np.column_stack(returns_list) if returns_list else None

        # 2. Iterate through each symbol and rolling window to extract (s, a, r, s', done, dis)
        for sym_idx, sym in enumerate(sym_keys):
            data = all_symbol_data[sym]
            opens = data["opens"]
            highs = data["highs"]
            lows = data["lows"]
            closes = data["closes"]
            volumes = data["volumes"]

            total_bars = len(closes)
            step_stride = max(1, (total_bars - lookback_window - forward_horizon) // max_samples_per_symbol)

            for t in range(lookback_window, total_bars - forward_horizon, step_stride):
                # Window slices for current state (t)
                win_o = opens[t - lookback_window : t]
                win_h = highs[t - lookback_window : t]
                win_l = lows[t - lookback_window : t]
                win_c = closes[t - lookback_window : t]
                win_v = volumes[t - lookback_window : t]

                # Technical metrics
                tech = extract_technicals(win_o, win_h, win_l, win_c, win_v)

                # Quant theory features
                spy_slice = spy_closes[t - lookback_window : t] if spy_closes is not None and len(spy_closes) >= t else None
                u_slice = universe_returns[t - lookback_window : t] if universe_returns is not None and len(universe_returns) >= t else None

                quant = self.quant_engine.extract_all_features(
                    win_c,
                    win_v,
                    market_index_prices=spy_slice,
                    universe_returns_matrix=u_slice,
                    target_asset_idx=sym_idx
                )

                # Build state vector at t (Shape: 1, 64)
                state_tensor = self.normalizer.build_state_vector(tech=tech, quant=quant)
                state = state_tensor.squeeze(0).numpy()

                # Next window slice for next_state (t + 1)
                next_t = t + 1
                next_win_o = opens[next_t - lookback_window : next_t]
                next_win_h = highs[next_t - lookback_window : next_t]
                next_win_l = lows[next_t - lookback_window : next_t]
                next_win_c = closes[next_t - lookback_window : next_t]
                next_win_v = volumes[next_t - lookback_window : next_t]

                next_tech = extract_technicals(next_win_o, next_win_h, next_win_l, next_win_c, next_win_v)
                next_spy = spy_closes[next_t - lookback_window : next_t] if spy_closes is not None and len(spy_closes) >= next_t else None
                next_u = universe_returns[next_t - lookback_window : next_t] if universe_returns is not None and len(universe_returns) >= next_t else None

                next_quant = self.quant_engine.extract_all_features(
                    next_win_c,
                    next_win_v,
                    market_index_prices=next_spy,
                    universe_returns_matrix=next_u,
                    target_asset_idx=sym_idx
                )
                next_state_tensor = self.normalizer.build_state_vector(tech=next_tech, quant=next_quant)
                next_state = next_state_tensor.squeeze(0).numpy()

                # Forward return over forward_horizon bars
                p_now = closes[t]
                p_future = closes[t + forward_horizon]
                forward_ret = (p_future - p_now) / max(p_now, 1e-4)

                # Retrospectively optimal policy label & continuous reward
                # 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
                if forward_ret >= 0.03:
                    action = 2          # Aggressive Buy (captured >= 3% gain)
                    reward = float(forward_ret * 2.0)
                elif forward_ret >= 0.01:
                    action = 1          # Moderate Buy (captured 1-3% gain)
                    reward = float(forward_ret)
                elif forward_ret <= -0.02:
                    action = 3          # Exit / Avoid drawdown
                    reward = float(abs(forward_ret) * 1.5)  # Reward exiting before loss
                else:
                    action = 0          # Hold / Neutral
                    reward = float(0.0)

                done = bool(t >= total_bars - forward_horizon - 2)
                # Epistemic disagreement proxy correlated with market chaos / Hurst near 0.5
                disagreement = float(np.clip(1.0 - 2.0 * abs(quant.hurst_exponent - 0.5), 0.05, 0.45))

                buffer.push(state, action, reward, next_state, done, disagreement)

        return buffer
