"""Unit tests for HFT Wick-Resistant Synthetic Stop-Loss Monitor.

Verifies:
1. Immunity to single-tick or two-tick predatory HFT wicks / bad prints.
2. Breach counter reset when price recovers above stop-loss level.
3. Stop-loss trigger confirmation upon reaching confirmation_ticks threshold.
4. Ratcheting stop-loss price upwards.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.governor import SyntheticStopLossMonitor


class TestSyntheticStopLossMonitor(unittest.TestCase):

    def setUp(self):
        self.monitor = SyntheticStopLossMonitor(confirmation_ticks=3)
        self.monitor.register_position(
            ticker="NVDA",
            entry_price=100.0,
            stop_loss_price=95.0
        )

    def test_single_tick_wick_immunity(self):
        # Tick 1: anomalous print dips to $94.50 (below $95.00)
        triggered, reason = self.monitor.evaluate_tick("NVDA", 94.50)
        self.assertFalse(triggered)
        self.assertIsNone(reason)

        # Tick 2: quick bounce back to $95.50 -> breach counter resets to 0
        triggered, reason = self.monitor.evaluate_tick("NVDA", 95.50)
        self.assertFalse(triggered)
        self.assertEqual(self.monitor._positions["NVDA"]["breaches"], 0)

        # Tick 3: another momentary wick dips to $94.80 -> breach counter is 1, not triggered
        triggered, reason = self.monitor.evaluate_tick("NVDA", 94.80)
        self.assertFalse(triggered)
        self.assertEqual(self.monitor._positions["NVDA"]["breaches"], 1)

    def test_three_tick_persistent_breach_triggers_stop(self):
        # Tick 1 below stop
        trig1, _ = self.monitor.evaluate_tick("NVDA", 94.90)
        self.assertFalse(trig1)

        # Tick 2 below stop
        trig2, _ = self.monitor.evaluate_tick("NVDA", 94.70)
        self.assertFalse(trig2)

        # Tick 3 below stop -> persistent breach confirmed!
        trig3, reason = self.monitor.evaluate_tick("NVDA", 94.50)
        self.assertTrue(trig3)
        self.assertIn("Synthetic stop-loss confirmed: 3 consecutive ticks below $95.00", reason)

    def test_stop_ratchet_upward(self):
        # Ratchet stop from $95.00 to $98.00
        self.monitor.update_stop_loss("NVDA", 98.00)
        self.assertEqual(self.monitor._positions["NVDA"]["stop_loss_price"], 98.00)

        # Price at $97.50 is now below new stop of $98.00
        trig1, _ = self.monitor.evaluate_tick("NVDA", 97.50)
        self.assertFalse(trig1)
        self.assertEqual(self.monitor._positions["NVDA"]["breaches"], 1)

    def test_unregister_position(self):
        self.monitor.unregister_position("NVDA")
        trig, reason = self.monitor.evaluate_tick("NVDA", 90.0)
        self.assertFalse(trig)
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
