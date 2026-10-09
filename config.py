"""Configuration management for Neuro-Symbolic Trading Bot.

Loads settings from .env file and environment variables with strict typing
and secret masking for audit logging.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(filepath: Path) -> None:
    """Manually parse .env file into os.environ if not already set."""
    if not filepath.exists():
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = 0.015       # 1.5% max capital risk per trade
    daily_drawdown_limit: float = 0.03     # 3% max portfolio drawdown halt
    emergency_liquidation: float = 0.05    # 5% max drawdown emergency close
    atr_multiplier_stop: float = 2.0       # 2x ATR trailing stop
    max_concurrent_positions: int = 5      # Max 5 active stocks
    min_market_cap_usd: float = 500_000_000.0  # $500M market cap floor
    min_avg_daily_volume_usd: float = 10_000_000.0  # $10M ADV floor
    min_stock_price: float = 5.0           # $5 price floor


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4                    # 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
    disagreement_threshold: float = 0.35    # Disagreement delta threshold tau
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    batch_size: int = 32
    replay_capacity: int = 10_000


@dataclass(frozen=True)
class ScoutConfig:
    max_discovered_candidates: int = 10
    user_agent: str = "TradingAgentsScout/1.0"
    jina_reader_prefix: str = "https://r.jina.ai/"
    scan_interval_seconds: int = 300


@dataclass(frozen=True)
class DaemonConfig:
    market_hours_interval: int = 60         # 60s sleep during active trading hours
    off_hours_sleep: int = 900              # 15m sleep when market is closed
    enable_post_market_training: bool = True


@dataclass(frozen=True)
class Config:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    paper_mode: bool
    risk: RiskConfig
    model: ModelConfig
    scout: ScoutConfig
    daemon: DaemonConfig = DaemonConfig()

    def mask_key(self, key: str) -> str:
        if not key or len(key) < 6:
            return "******"
        return f"{key[:4]}...{key[-4:]}"

    def summary(self) -> str:
        return (
            f"[Config] Paper Mode: {self.paper_mode} | "
            f"Alpaca Base: {self.alpaca_base_url} | "
            f"API Key: {self.mask_key(self.alpaca_api_key)} | "
            f"Max Risk/Trade: {self.risk.max_risk_per_trade * 100:.1f}% | "
            f"Daily Stop: {self.risk.daily_drawdown_limit * 100:.1f}%"
        )


def load_config(env_path: str | Path | None = None) -> Config:
    """Load and validate configuration from .env and environment."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    else:
        env_path = Path(env_path)

    _load_dotenv(env_path)

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper_mode = os.getenv("PAPER_MODE", "true").lower() in ("true", "1", "yes")

    risk_cfg = RiskConfig(
        max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", 0.015)),
        daily_drawdown_limit=float(os.getenv("DAILY_DRAWDOWN_LIMIT", 0.03)),
        atr_multiplier_stop=float(os.getenv("ATR_MULTIPLIER_STOP", 2.0)),
    )

    model_cfg = ModelConfig(
        disagreement_threshold=float(os.getenv("DISAGREEMENT_THRESHOLD", 0.35)),
    )

    scout_cfg = ScoutConfig()
    daemon_cfg = DaemonConfig(
        market_hours_interval=int(os.getenv("DAEMON_INTERVAL_MARKET_HOURS", 60)),
        off_hours_sleep=int(os.getenv("DAEMON_INTERVAL_OFF_HOURS", 900)),
    )

    return Config(
        alpaca_api_key=api_key,
        alpaca_secret_key=secret_key,
        alpaca_base_url=base_url,
        paper_mode=paper_mode,
        risk=risk_cfg,
        model=model_cfg,
        scout=scout_cfg,
        daemon=daemon_cfg,
    )
