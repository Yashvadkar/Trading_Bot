import unittest
from unittest.mock import MagicMock, patch
from broker.order_fsm import OrderLifecycleFSM, OrderState
from broker.alpaca_client import AlpacaPaperClient
from config import load_config


class TestOrderLifecycleFSM(unittest.TestCase):
    def setUp(self):
        self.fsm = OrderLifecycleFSM()

    def test_valid_monotonic_progression(self):
        cid = "bot_test_order_001"
        self.fsm.register_order(cid, initial_state=OrderState.PENDING_NEW, sequence_id=1)
        self.assertEqual(self.fsm.get_state(cid), OrderState.PENDING_NEW)

        # Transition to NEW
        self.assertTrue(self.fsm.transition(cid, OrderState.NEW, sequence_id=2))
        self.assertEqual(self.fsm.get_state(cid), OrderState.NEW)

        # Transition to PARTIALLY_FILLED
        self.assertTrue(self.fsm.transition(cid, OrderState.PARTIALLY_FILLED, sequence_id=3, filled_qty=10))
        self.assertEqual(self.fsm.get_state(cid), OrderState.PARTIALLY_FILLED)

        # Transition to FILLED
        self.assertTrue(self.fsm.transition(cid, OrderState.FILLED, sequence_id=4, filled_qty=50))
        self.assertEqual(self.fsm.get_state(cid), OrderState.FILLED)
        self.assertTrue(self.fsm.is_terminal(cid))

    def test_terminal_state_immutability(self):
        cid = "bot_test_order_terminal"
        self.fsm.register_order(cid, initial_state=OrderState.NEW)
        self.fsm.transition(cid, OrderState.FILLED)

        # Out-of-order CANCELED or PARTIALLY_FILLED must be rejected
        self.assertFalse(self.fsm.transition(cid, OrderState.CANCELED))
        self.assertFalse(self.fsm.transition(cid, OrderState.PARTIALLY_FILLED))
        self.assertEqual(self.fsm.get_state(cid), OrderState.FILLED)

    def test_sequence_gated_lww(self):
        cid = "bot_test_seq"
        self.fsm.register_order(cid, initial_state=OrderState.NEW, sequence_id=10)

        # Older sequence message arriving late must be rejected
        self.assertFalse(self.fsm.transition(cid, OrderState.PARTIALLY_FILLED, sequence_id=5))
        self.assertEqual(self.fsm.get_state(cid), OrderState.NEW)

        # Newer sequence accepted
        self.assertTrue(self.fsm.transition(cid, OrderState.PARTIALLY_FILLED, sequence_id=12))
        self.assertEqual(self.fsm.get_state(cid), OrderState.PARTIALLY_FILLED)

    def test_in_flight_modification_lock(self):
        cid = "bot_test_lock"
        self.fsm.register_order(cid, initial_state=OrderState.NEW)

        # First lock succeeds
        self.assertTrue(self.fsm.acquire_modification_lock(cid))
        # Concurrent lock attempt fails
        self.assertFalse(self.fsm.acquire_modification_lock(cid))

        # Releasing allows locking again
        self.fsm.release_modification_lock(cid)
        self.assertTrue(self.fsm.acquire_modification_lock(cid))
        self.fsm.release_modification_lock(cid)

    def test_alpaca_race_condition_reconciliation(self):
        cfg = load_config()
        client = AlpacaPaperClient(cfg, dry_run=False)

        # Mock GET open orders returning an order
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = [{"id": "ord_race_123", "type": "stop", "stop_price": "140.0"}]

        # Mock PATCH returning 422 Unprocessable Entity (order filled milliseconds prior)
        mock_patch_resp = MagicMock()
        mock_patch_resp.status_code = 422
        mock_patch_resp.text = "order already closed"

        with patch.object(client.session, "get", return_value=mock_get_resp), \
             patch.object(client.session, "patch", return_value=mock_patch_resp):
            success = client.update_stop_loss("AAPL", 145.0)
            self.assertTrue(success)
            self.assertEqual(client.fsm.get_state("ord_race_123"), OrderState.FILLED)

    def test_force_fsm_reconciliation_direct(self):
        cid = "bot_force_test"
        self.fsm.register_order(cid, initial_state=OrderState.NEW)
        self.fsm.acquire_modification_lock(cid)
        # Forced reconciliation should override lock and transition to FILLED
        self.assertTrue(self.fsm.force_fsm_reconciliation(cid, OrderState.FILLED))
        self.assertEqual(self.fsm.get_state(cid), OrderState.FILLED)
        self.assertTrue(self.fsm.is_terminal(cid))

    def test_alpaca_update_stop_loss_pessimistic_terminal_skip(self):
        cfg = load_config()
        client = AlpacaPaperClient(cfg, dry_run=False)
        order_id = "ord_terminal_skip"
        client.fsm.register_order(order_id, initial_state=OrderState.FILLED)

        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = [{"id": order_id, "type": "stop", "stop_price": "140.0"}]

        with patch.object(client.session, "get", return_value=mock_get_resp), \
             patch.object(client.session, "patch") as mock_patch:
            # Because order is already terminal, patch must NOT be called
            success = client.update_stop_loss("AAPL", 145.0)
            self.assertFalse(success)
            mock_patch.assert_not_called()

    def test_alpaca_404_reconciliation(self):
        cfg = load_config()
        client = AlpacaPaperClient(cfg, dry_run=False)

        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = [{"id": "ord_404_test", "type": "stop", "stop_price": "140.0"}]

        mock_patch_resp = MagicMock()
        mock_patch_resp.status_code = 404
        mock_patch_resp.text = "order not found"

        with patch.object(client.session, "get", return_value=mock_get_resp), \
             patch.object(client.session, "patch", return_value=mock_patch_resp):
            success = client.update_stop_loss("AAPL", 145.0)
            self.assertTrue(success)
            self.assertEqual(client.fsm.get_state("ord_404_test"), OrderState.FILLED)


if __name__ == "__main__":
    unittest.main()

