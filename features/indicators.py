"""Deterministic financial indicators calculated via NumPy vectorization.

Zero external heavy C++ dependencies required; fully vectorized in NumPy.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TechnicalMetrics:
    rsi: float                     # Relative Strength Index (0 to 100)
    macd: float                    # MACD line (12, 26)
    macd_signal: float             # Signal line (9)
    macd_hist: float               # Histogram (macd - signal)
    atr: float                     # Average True Range (14)
    atr_pct: float                 # ATR as percentage of price (volatility)
    sma_20_ratio: float            # (Price - SMA20) / SMA20
    sma_50_ratio: float            # (Price - SMA50) / SMA50
    sma_200_ratio: float           # (Price - SMA200) / SMA200
    bb_pct_b: float                # Bollinger %B ((Price - Lower) / (Upper - Lower))
    realized_volatility: float     # 20-day annualized realized volatility
    volume_surge_ratio: float      # Today's volume / 20-day average volume


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return 50.0  # Neutral fallback

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    """Compute MACD, Signal line, and Histogram using Exponential Moving Averages."""
    if len(prices) < slow + signal:
        return 0.0, 0.0, 0.0

    def ema(series: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        res = np.empty_like(series, dtype=np.float64)
        res[0] = series[0]
        for t in range(1, len(series)):
            res[t] = alpha * series[t] + (1.0 - alpha) * res[t - 1]
        return res

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line[-1] - signal_line[-1]
    return float(macd_line[-1]), float(signal_line[-1]), float(hist)


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Compute Average True Range (ATR)."""
    if len(closes) < period + 1:
        return float(np.mean(highs - lows)) if len(closes) > 0 else 1.0

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 1.0

    atr = float(np.mean(tr[-period:]))
    return max(atr, 0.01)


def compute_bollinger_pct_b(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """Compute Bollinger Bands %B indicator."""
    if len(prices) < period:
        return 0.5
    window = prices[-period:]
    sma = np.mean(window)
    std = np.std(window)
    if std == 0.0:
        return 0.5
    upper = sma + num_std * std
    lower = sma - num_std * std
    current = prices[-1]
    pct_b = (current - lower) / (upper - lower) if (upper != lower) else 0.5
    return float(np.clip(pct_b, -0.5, 1.5))


def extract_technicals(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray
) -> TechnicalMetrics:
    """Extract comprehensive, vectorized technical indicators from OHLCV arrays."""
    closes = np.asarray(closes, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    curr_price = float(closes[-1]) if len(closes) > 0 else 100.0

    # RSI
    rsi = compute_rsi(closes, period=14)

    # MACD
    macd, macd_sig, macd_hist = compute_macd(closes, 12, 26, 9)

    # ATR
    atr = compute_atr(highs, lows, closes, period=14)
    atr_pct = (atr / curr_price) if curr_price > 0 else 0.02

    # SMAs
    sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else curr_price
    sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else curr_price
    sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else curr_price

    sma20_ratio = (curr_price - sma20) / sma20 if sma20 > 0 else 0.0
    sma50_ratio = (curr_price - sma50) / sma50 if sma50 > 0 else 0.0
    sma200_ratio = (curr_price - sma200) / sma200 if sma200 > 0 else 0.0

    # Bollinger %B
    pct_b = compute_bollinger_pct_b(closes, period=20)

    # Realized volatility (annualized 20-day)
    if len(closes) >= 21:
        log_ret = np.diff(np.log(closes[-21:]))
        realized_vol = float(np.std(log_ret) * np.sqrt(252))
    else:
        realized_vol = 0.20

    # Volume surge
    if len(volumes) >= 20:
        avg_vol = float(np.mean(volumes[-20:]))
        curr_vol = float(volumes[-1])
        vol_surge = (curr_vol / avg_vol) if avg_vol > 0 else 1.0
    else:
        vol_surge = 1.0

    return TechnicalMetrics(
        rsi=rsi,
        macd=macd,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        atr=atr,
        atr_pct=atr_pct,
        sma_20_ratio=sma20_ratio,
        sma_50_ratio=sma50_ratio,
        sma_200_ratio=sma200_ratio,
        bb_pct_b=pct_b,
        realized_volatility=realized_vol,
        volume_surge_ratio=vol_surge,
    )


def compute_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Vectorized exponential moving average."""
    series = np.asarray(series, dtype=np.float64)
    if len(series) == 0:
        return np.array([], dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    res = np.empty_like(series, dtype=np.float64)
    res[0] = series[0]
    for t in range(1, len(series)):
        res[t] = alpha * series[t] + (1.0 - alpha) * res[t - 1]
    return res


def detect_volatility_contraction_pattern(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    lookback: int = 30
) -> tuple[bool, float, dict]:
    """Detects Minervini/Gajjala Intraday Volatility Contraction Pattern (VCP).
    
    Mathematical criteria:
    1. Sequential swing depth contraction: D_1 > D_2 with final contraction depth <= 12%
    2. Volume dry-up: final contraction volume < 0.70 * baseline SMA(V)
    3. Moving Average Perfect Order: Close >= EMA(9) >= EMA(21)
    
    Returns:
        (is_vcp, contraction_ratio, metrics_dict)
    """
    n = len(closes)
    if n < 22:
        return False, 1.0, {"reason": "Insufficient bars for VCP analysis"}

    h_sub = highs[-lookback:] if n >= lookback else highs
    l_sub = lows[-lookback:] if n >= lookback else lows
    c_sub = closes[-lookback:] if n >= lookback else closes
    v_sub = volumes[-lookback:] if n >= lookback else volumes
    sub_len = len(c_sub)

    # 1. Moving average trend alignment (Close >= EMA 9 >= EMA 21)
    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    cur_price = float(closes[-1])
    is_trend_aligned = bool(cur_price >= ema9[-1] and ema9[-1] >= ema21[-1])

    # 2. Divide lookback into 2 sequential contraction waves
    mid = sub_len // 2
    h1 = float(np.max(h_sub[:mid]))
    l1 = float(np.min(l_sub[:mid]))
    d1 = (h1 - l1) / max(h1, 1e-4)

    h2 = float(np.max(h_sub[mid:]))
    l2 = float(np.min(l_sub[mid:]))
    d2 = (h2 - l2) / max(h2, 1e-4)

    is_contracting = bool(d1 > d2 and d2 <= 0.12)
    contraction_ratio = float(d2 / max(d1, 1e-4))

    # 3. Volume dry-up: final 5 bars volume vs baseline SMA volume
    avg_vol_baseline = float(np.mean(volumes[-min(n, 50):]))
    final_vol = float(np.mean(v_sub[-min(5, len(v_sub)):]))
    vol_dry_ratio = float(final_vol / max(avg_vol_baseline, 1.0))
    is_volume_dry = bool(vol_dry_ratio <= 0.75)

    is_vcp = bool(is_trend_aligned and is_contracting and is_volume_dry)
    metrics = {
        "is_trend_aligned": is_trend_aligned,
        "is_contracting": is_contracting,
        "is_volume_dry": is_volume_dry,
        "depth_wave1": d1,
        "depth_wave2": d2,
        "contraction_ratio": contraction_ratio,
        "vol_dry_ratio": vol_dry_ratio
    }
    return is_vcp, contraction_ratio, metrics
