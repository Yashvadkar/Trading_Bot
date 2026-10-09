"""SQLite Audit Ledger & Telemetry Database (Backend Systems & Data Architecture).

Tracks every discovery event, neural debate score, order directive,
and executed trade with parameterized SQL statements and clean DAO patterns.
"""

from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TradeRecord:
    ticker: str
    side: str
    shares: int
    price: float
    stop_loss: float
    order_id: str
    status: str
    realized_pnl: float = 0.0


class TradeLedger:
    """Manages SQLite persistent audit logging for the trading bot."""

    def __init__(self, db_path: str | Path = "trades.db"):
        self.db_path = Path(db_path)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create structured tables with proper indexes."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS discovery_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    mention_count INTEGER NOT NULL,
                    sentiment_score REAL NOT NULL,
                    catalysts TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_disc_ticker ON discovery_log(ticker, timestamp DESC);

                CREATE TABLE IF NOT EXISTS debate_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    bull_conf REAL NOT NULL,
                    bear_conf REAL NOT NULL,
                    disagreement_var REAL NOT NULL,
                    chosen_action INTEGER NOT NULL,
                    is_approved INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_deb_ticker ON debate_log(ticker, timestamp DESC);

                CREATE TABLE IF NOT EXISTS trades_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    shares INTEGER NOT NULL,
                    price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    realized_pnl REAL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_trade_ticker ON trades_log(ticker, timestamp DESC);
            """)

    def log_discovery(self, ticker: str, mention_count: int, sentiment_score: float, catalysts: list[str]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO discovery_log (timestamp, ticker, mention_count, sentiment_score, catalysts) VALUES (?, ?, ?, ?, ?)",
                (ts, ticker, mention_count, sentiment_score, ",".join(catalysts))
            )

    def log_debate(
        self,
        ticker: str,
        bull_conf: float,
        bear_conf: float,
        disagreement_var: float,
        action: int,
        is_approved: bool
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO debate_log (timestamp, ticker, bull_conf, bear_conf, disagreement_var, chosen_action, is_approved) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, ticker, bull_conf, bear_conf, disagreement_var, action, 1 if is_approved else 0)
            )

    def log_trade(self, record: TradeRecord) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO trades_log (timestamp, ticker, side, shares, price, stop_loss, order_id, status, realized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, record.ticker, record.side, record.shares, record.price, record.stop_loss, record.order_id, record.status, record.realized_pnl)
            )

    def get_recent_trades(self, limit: int = 10) -> list[dict]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM trades_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
