"""SQLite Audit Ledger & Telemetry Database (Backend Systems & Data Architecture).

Tracks every discovery event, neural debate score, order directive,
and executed trade with parameterized SQL statements and clean DAO patterns.
"""

from __future__ import annotations
import sqlite3
import asyncio
from contextlib import contextmanager
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

    @contextmanager
    def _get_connection(self):
        """Yields an active SQLite connection configured with WAL pragmas, committing and closing on exit."""
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _sync_batch_write(self, queries: list[tuple[str, tuple]]) -> None:
        """Strictly synchronous atomic batch write executed within worker thread or caller.
        
        Uses BEGIN IMMEDIATE to acquire an immediate write lock, commits on success,
        rolls back on error, and ensures PRAGMA wal_checkpoint(PASSIVE) in finally block.
        """
        if not queries:
            return
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None  # Explicit manual transaction control
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        try:
            conn.execute("BEGIN IMMEDIATE;")
            for sql, params in queries:
                conn.execute(sql, params)
            conn.execute("COMMIT;")
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            except Exception:
                pass
            conn.close()

    async def async_commit_state(self, queries: list[tuple[str, tuple]]) -> None:
        """Asynchronously offloads batch write to thread pool executor, maintaining zero-await transaction boundaries."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_batch_write, queries)

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
        query = (
            "INSERT INTO discovery_log (timestamp, ticker, mention_count, sentiment_score, catalysts) VALUES (?, ?, ?, ?, ?)",
            (ts, ticker, mention_count, sentiment_score, ",".join(catalysts))
        )
        self._sync_batch_write([query])

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
        query = (
            "INSERT INTO debate_log (timestamp, ticker, bull_conf, bear_conf, disagreement_var, chosen_action, is_approved) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, ticker, bull_conf, bear_conf, disagreement_var, action, 1 if is_approved else 0)
        )
        self._sync_batch_write([query])

    def log_trade(self, record: TradeRecord) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        query = (
            "INSERT INTO trades_log (timestamp, ticker, side, shares, price, stop_loss, order_id, status, realized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, record.ticker, record.side, record.shares, record.price, record.stop_loss, record.order_id, record.status, record.realized_pnl)
        )
        self._sync_batch_write([query])

    def get_recent_trades(self, limit: int = 10) -> list[dict]:
        """Eagerly drains recent trade records into plain memory dicts to avoid lingering cursor locks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades_log ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def checkpoint_wal(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        """Executes explicit WAL checkpoint to truncate WAL file and reclaim disk frames.
        
        Mitigates checkpoint starvation caused by high-throughput concurrent readers.
        Modes: PASSIVE, FULL, RESTART, TRUNCATE.
        Returns: (busy, log_frames, checkpointed_frames)
        """
        valid_mode = mode.upper() if mode.upper() in ("PASSIVE", "FULL", "RESTART", "TRUNCATE") else "PASSIVE"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA wal_checkpoint({valid_mode});")
            row = cursor.fetchone()
            if row:
                return int(row[0]), int(row[1]), int(row[2])
            return 0, 0, 0
