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


if __name__ == "__main__":
    unittest.main()
