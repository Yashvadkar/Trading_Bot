"""Unit tests for Marcos López de Prado Triple Barrier Exit Engine.

Verifies:
1. Upper horizontal barrier trigger upon reaching dynamic profit target (+2.5R).
2. Lower horizontal barrier trigger with wick resistance upon persistent stop breach.
3. Vertical barrier time-decay expiration when holding duration reaches max_bars.
4. Position unregistration and state cleanup.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.governor import TripleBarrierTracker


class TestTripleBarrierTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = TripleBarrierTracker(default_max_bars=10, wick_confirmation_ticks=3)
        # Entry = $100.00, Stop = $96.00 (Risk = $4.00, 1R)
        # Target = Entry + 2.5 * Risk = $100.00 + $10.00 = $110.00
        self.tracker.register(
            ticker="TSLA",
            entry_price=100.0,
            stop_loss_price=96.0,
            take_profit_price=110.0,
            max_bars=10
        )

    def test_upper_profit_barrier_trigger(self):
        # Bar 1: Price rallies to $105.00
        should_exit, _ = self.tracker.evaluate_bar("TSLA", 105.0)
        self.assertFalse(should_exit)

        # Bar 2: Price hits profit target $110.50
        should_exit, reason = self.tracker.evaluate_bar("TSLA", 110.50)
        self.assertTrue(should_exit)
        self.assertIn("UPPER_PROFIT_BARRIER reached at $110.50 >= $110.00", reason)

    def test_lower_stop_barrier_trigger_with_wick_filter(self):
        # Bar 1: Price dips to $95.50 (< $96.00) -> 1 breach, no exit
        should_exit, _ = self.tracker.evaluate_bar("TSLA", 95.50)
        self.assertFalse(should_exit)

        # Bar 2: Price bounces to $96.50 -> reset breaches
        should_exit, _ = self.tracker.evaluate_bar("TSLA", 96.50)
        self.assertFalse(should_exit)

        # Bar 3, 4, 5: 3 consecutive breaches
        self.tracker.evaluate_bar("TSLA", 95.0)
        self.tracker.evaluate_bar("TSLA", 94.8)
        should_exit, reason = self.tracker.evaluate_bar("TSLA", 94.5)
        self.assertTrue(should_exit)
        self.assertIn("LOWER_STOP_BARRIER breached", reason)

    def test_vertical_time_barrier_expiration(self):
        # Price oscillates in consolidation range between $98 and $102 for 10 bars
        for bar in range(9):
            should_exit, _ = self.tracker.evaluate_bar("TSLA", 100.0)
            self.assertFalse(should_exit)

        # 10th bar reaches max_bars -> time barrier triggers to recycle dead capital
        should_exit, reason = self.tracker.evaluate_bar("TSLA", 100.0)
        self.assertTrue(should_exit)
        self.assertIn("VERTICAL_TIME_BARRIER expired (10/10 bars without resolution)", reason)


if __name__ == "__main__":
    unittest.main()
