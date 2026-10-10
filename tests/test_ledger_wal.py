import unittest
import tempfile
import threading
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker.ledger import TradeLedger, TradeRecord


class TestLedgerWAL(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_trades.db"
        self.ledger = TradeLedger(db_path=self.db_path)

    def tearDown(self):
        try:
            self.ledger.stop_checkpoint_daemon()
            self.temp_dir.cleanup()
        except Exception:
            pass


    def test_wal_mode_active(self):
        with self.ledger._get_connection() as conn:
            row = conn.execute("PRAGMA journal_mode;").fetchone()
            self.assertEqual(row[0].lower(), "wal")

    def test_eager_cursor_draining(self):
        for i in range(5):
            rec = TradeRecord(
                ticker=f"SYM{i}",
                side="buy",
                shares=10 * (i + 1),
                price=100.0 + i,
                stop_loss=95.0,
                order_id=f"ord_{i}",
                status="filled",
                realized_pnl=0.0
            )
            self.ledger.log_trade(rec)

        trades = self.ledger.get_recent_trades(limit=10)
        self.assertEqual(len(trades), 5)
        self.assertIsInstance(trades[0], dict)
        self.assertEqual(trades[0]["ticker"], "SYM4")

    def test_checkpoint_wal(self):
        # Insert records to generate WAL frames
        for i in range(20):
            self.ledger.log_discovery(f"TICK{i}", 5, 0.8, ["earnings", "growth"])

        # Execute checkpoint
        busy, log, checkpointed = self.ledger.checkpoint_wal(mode="PASSIVE")
        self.assertEqual(busy, 0)
        self.assertGreaterEqual(checkpointed, 0)

    def test_concurrent_multithreaded_access(self):
        errors = []

        def writer_task(thread_id):
            try:
                for i in range(15):
                    rec = TradeRecord(
                        ticker=f"T{thread_id}_{i}",
                        side="buy",
                        shares=5,
                        price=50.0,
                        stop_loss=45.0,
                        order_id=f"t{thread_id}_{i}",
                        status="filled"
                    )
                    self.ledger.log_trade(rec)
            except Exception as e:
                errors.append(e)

        def reader_task():
            try:
                for _ in range(15):
                    _ = self.ledger.get_recent_trades(limit=5)
            except Exception as e:
                errors.append(e)

        threads = []
        for tid in range(4):
            threads.append(threading.Thread(target=writer_task, args=(tid,)))
            threads.append(threading.Thread(target=reader_task))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread concurrency errors encountered: {errors}")

    def test_sync_batch_write_atomicity(self):
        queries = [
            ("INSERT INTO discovery_log (timestamp, ticker, mention_count, sentiment_score, catalysts) VALUES (?, ?, ?, ?, ?)",
             ("2026-10-10T00:00:00Z", f"BATCH_{i}", 10, 0.9, "momentum"))
            for i in range(5)
        ]
        self.ledger._sync_batch_write(queries)
        with self.ledger._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM discovery_log WHERE ticker LIKE 'BATCH_%';").fetchone()[0]
            self.assertEqual(count, 5)

    def test_sync_batch_write_rollback_on_error(self):
        queries = [
            ("INSERT INTO discovery_log (timestamp, ticker, mention_count, sentiment_score, catalysts) VALUES (?, ?, ?, ?, ?)",
             ("2026-10-10T00:00:00Z", "ROLLBACK_TEST", 10, 0.9, "momentum")),
            ("INSERT INTO non_existent_table (col) VALUES (?)", ("error_val",))
        ]
        with self.assertRaises(Exception):
            self.ledger._sync_batch_write(queries)

        with self.ledger._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM discovery_log WHERE ticker = 'ROLLBACK_TEST';").fetchone()[0]
            self.assertEqual(count, 0)

    def test_async_commit_state(self):
        import asyncio

        async def run_async():
            queries = [
                ("INSERT INTO discovery_log (timestamp, ticker, mention_count, sentiment_score, catalysts) VALUES (?, ?, ?, ?, ?)",
                 ("2026-10-10T00:00:00Z", f"ASYNC_{i}", 15, 0.95, "breakout"))
                for i in range(3)
            ]
            await self.ledger.async_commit_state(queries)

        asyncio.run(run_async())

        with self.ledger._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM discovery_log WHERE ticker LIKE 'ASYNC_%';").fetchone()[0]
            self.assertEqual(count, 3)

    def test_checkpoint_daemon_lifecycle(self):
        # Start daemon with short interval
        self.ledger.start_checkpoint_daemon(interval_sec=0.1)
        self.assertIsNotNone(self.ledger._checkpoint_thread)
        self.assertTrue(self.ledger._checkpoint_thread.is_alive())

        # Calling start again is idempotent
        self.ledger.start_checkpoint_daemon(interval_sec=0.1)
        self.assertTrue(self.ledger._checkpoint_thread.is_alive())

        # Stop daemon terminates thread cleanly
        self.ledger.stop_checkpoint_daemon(timeout=1.0)
        self.assertIsNone(self.ledger._checkpoint_thread)

    def test_restart_wal_checkpoint(self):
        for i in range(10):
            self.ledger.log_discovery(f"RESTART_{i}", 3, 0.7, ["catalyst"])
        busy, log_frames, checkpointed = self.ledger.checkpoint_wal(mode="RESTART")
        self.assertEqual(busy, 0)
        self.assertGreaterEqual(checkpointed, 0)


if __name__ == "__main__":
    unittest.main()


