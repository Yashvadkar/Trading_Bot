import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.historical_loader import HistoricalDataLoader, TemporalFiltrationError, PITDatum


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

    def test_pit_datum_operational_availability_filtration(self):
        # Decision boundary at t = 1000.0
        t_decision = 1000.0

        data = [
            PITDatum(event_timestamp=950.0, available_timestamp=960.0, value=100.5),
            PITDatum(event_timestamp=980.0, available_timestamp=995.0, value=101.2),
            # Event occurred in past (990.0) but reporting was delayed and not available until 1005.0 > t
            PITDatum(event_timestamp=990.0, available_timestamp=1005.0, value=102.0),
            # Event occurred in future
            PITDatum(event_timestamp=1010.0, available_timestamp=1020.0, value=103.5),
        ]

        # Filtering must retain strictly records with available_timestamp <= t_decision
        filtered = HistoricalDataLoader.filter_pit_sequence(data, decision_timestamp=t_decision)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].value, 100.5)
        self.assertEqual(filtered[1].value, 101.2)

        # assert_pit_filtration succeeds on filtered data
        self.assertTrue(HistoricalDataLoader.assert_pit_filtration(filtered, decision_timestamp=t_decision))

        # assert_pit_filtration raises TemporalFiltrationError on unfiltered data containing lookahead
        with self.assertRaises(TemporalFiltrationError):
            HistoricalDataLoader.assert_pit_filtration(data, decision_timestamp=t_decision)


if __name__ == "__main__":
    unittest.main()

