import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.historical_loader import HistoricalDataLoader, TemporalFiltrationError


class TestPITFiltration(unittest.TestCase):

    def test_natural_filtration_verification(self):
        # Case 1: Strictly past data available at decision time t (I_t subset of F_t)
        self.assertTrue(
            HistoricalDataLoader.verify_natural_filtration(decision_index=100, data_max_index=99)
        )

        # Case 2: Temporal violation where feature ingests bar at decision index t or beyond
        with self.assertRaises(TemporalFiltrationError):
            HistoricalDataLoader.verify_natural_filtration(decision_index=100, data_max_index=100)

        with self.assertRaises(TemporalFiltrationError):
            HistoricalDataLoader.verify_natural_filtration(decision_index=100, data_max_index=105)

    def test_constituent_inception_tracking(self):
        inceptions = HistoricalDataLoader.CONSTITUENT_INCEPTION
        self.assertIn("SPY", inceptions)
        self.assertIn("NVDA", inceptions)
        self.assertEqual(inceptions["NVDA"], "1999-01-22")
        self.assertEqual(inceptions["TSLA"], "2010-06-29")


if __name__ == "__main__":
    unittest.main()
