"""Order Lifecycle Finite State Machine (FSM) & Event Sequencer.

Provides strict monotonic state progression, terminal state immutability,
and sequence-gated event replication (LWW CRDT) to eliminate REST vs WebSocket
race conditions in high-throughput algorithmic execution.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from enum import Enum


class OrderState(str, Enum):
    PENDING_NEW = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED)


@dataclass
class OrderStateRecord:
    client_order_id: str
    exchange_order_id: str | None
    state: OrderState
    sequence_id: int
    last_updated: float
    filled_qty: int = 0
    in_flight_modification: bool = False
    last_event_id: str | None = None



class OrderLifecycleFSM:
    """Manages order state lifecycles with monotonic gating and sequence guards."""

    # Set of permitted directed state transitions
    VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
        OrderState.PENDING_NEW: {
            OrderState.NEW,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
        },
        OrderState.NEW: {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
        },
        OrderState.PARTIALLY_FILLED: {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
        },
        # Terminal states are strictly immutable:
        OrderState.FILLED: set(),
        OrderState.CANCELED: set(),
        OrderState.REJECTED: set(),
    }

    def __init__(self):
        self._orders: dict[str, OrderStateRecord] = {}
        self._exchange_to_client: dict[str, str] = {}

    def register_order(
        self,
        client_order_id: str,
        exchange_order_id: str | None = None,
        initial_state: OrderState = OrderState.PENDING_NEW,
        sequence_id: int = 0
    ) -> OrderStateRecord:
        """Initializes a new order in the state machine."""
        now = time.time()
        record = OrderStateRecord(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            state=initial_state,
            sequence_id=sequence_id,
            last_updated=now
        )
        self._orders[client_order_id] = record
        if exchange_order_id:
            self._exchange_to_client[exchange_order_id] = client_order_id
        return record

    def transition(
        self,
        client_order_id: str,
        new_state: OrderState,
        sequence_id: int = 0,
        timestamp: float | None = None,
        filled_qty: int = 0
    ) -> bool:
        """Attempts to transition an order to new_state with monotonic and sequence checks.
        
        Returns:
            True if the transition was applied, False if rejected (e.g. out-of-order or terminal).
        """
        now = timestamp if timestamp is not None else time.time()
        record = self._orders.get(client_order_id)
        if not record:
            # Register lazily if arriving from external exchange stream
            self.register_order(client_order_id, initial_state=new_state, sequence_id=sequence_id)
            return True

        # 1. Monotonic Terminal Guard: Terminal states can never be reverted
        if record.state.is_terminal:
            return False

        # 2. Sequence Gate (LWW): Reject older sequence events if sequence numbers are provided
        if sequence_id > 0 and record.sequence_id > 0 and sequence_id <= record.sequence_id:
            return False

        # 3. Valid Transition Check
        allowed = self.VALID_TRANSITIONS.get(record.state, set())
        if new_state not in allowed and new_state != record.state:
            return False

        # Apply valid transition
        record.state = new_state
        record.sequence_id = max(record.sequence_id, sequence_id)
        record.last_updated = now
        if filled_qty > 0:
            record.filled_qty = filled_qty
        if new_state.is_terminal:
            record.in_flight_modification = False

        return True

    def acquire_modification_lock(self, client_order_id: str) -> bool:
        """Pessimistic lock: check if order is resting and can safely be patched."""
        record = self._orders.get(client_order_id)
        if not record:
            return True  # If not tracked yet, allow with caution
        if record.state.is_terminal:
            return False
        if record.in_flight_modification:
            return False
        record.in_flight_modification = True
        return True

    def release_modification_lock(self, client_order_id: str) -> None:
        """Releases the modification lock for client_order_id."""
        record = self._orders.get(client_order_id)
        if record:
            record.in_flight_modification = False

    def get_state(self, identifier: str) -> OrderState | None:
        """Retrieves state by either client_order_id or exchange_order_id."""
        if identifier in self._orders:
            return self._orders[identifier].state
        client_id = self._exchange_to_client.get(identifier)
        if client_id and client_id in self._orders:
            return self._orders[client_id].state
        return None

    def is_terminal(self, identifier: str) -> bool:
        state = self.get_state(identifier)
        return state.is_terminal if state else False

    def map_exchange_id(self, client_order_id: str, exchange_order_id: str) -> None:
        self._exchange_to_client[exchange_order_id] = client_order_id
        if client_order_id in self._orders:
            self._orders[client_order_id].exchange_order_id = exchange_order_id

    def force_fsm_reconciliation(
        self,
        identifier: str,
        new_state: OrderState = OrderState.FILLED,
        reason: str = "matching_engine_reconciliation"
    ) -> bool:
        """Forcibly transitions an order to a terminal state (default: FILLED).
        
        Invoked when the exchange returns 404 (not found / purged) or 422 (already filled/closed).
        Bypasses non-terminal state transition checks and clears in-flight modification locks.
        """
        record = self._orders.get(identifier)
        if not record:
            client_id = self._exchange_to_client.get(identifier)
            if client_id:
                record = self._orders.get(client_id)

        now = time.time()
        if not record:
            self.register_order(identifier, initial_state=new_state)
            return True

        record.state = new_state
        record.in_flight_modification = False
        record.last_updated = now
        return True

    def process_sse_event(
        self,
        client_order_id: str,
        new_state: OrderState,
        event_id: str,
        filled_qty: int = 0,
        exchange_order_id: str | None = None
    ) -> bool:
        """Processes Alpaca Activity SSE events using ULID lexicographical sorting 
        to guarantee state machine determinism and Last-Write-Wins (CRDT) consistency.
        
        Returns True if transition applied, False if rejected as stale or terminal reversion.
        """
        now = time.time()
        record = self._orders.get(client_order_id)
        if not record and exchange_order_id:
            c_id = self._exchange_to_client.get(exchange_order_id)
            if c_id:
                record = self._orders.get(c_id)

        # 1. New order ingestion
        if not record:
            new_rec = self.register_order(client_order_id, exchange_order_id=exchange_order_id, initial_state=new_state)
            new_rec.last_event_id = event_id
            if filled_qty > 0:
                new_rec.filled_qty = filled_qty
            return True

        # 2. Sequence Gate: Reject stale, out-of-order messages from the thread pool.
        # ULIDs are lexicographically sortable by high-precision timestamp.
        if record.last_event_id and event_id <= record.last_event_id:
            return False  # Discard out-of-order event

        # 3. Immutability Guard: Terminal states can never be reverted
        if record.state.is_terminal:
            return False  # Order is already finalized, discard delayed transition

        # 4. Valid Transition Check
        allowed = self.VALID_TRANSITIONS.get(record.state, set())
        if new_state not in allowed and new_state != record.state:
            return False

        # 5. Monotonic Update
        record.state = new_state
        record.last_event_id = event_id
        record.last_updated = now
        if filled_qty > 0:
            record.filled_qty = filled_qty
        if new_state.is_terminal:
            record.in_flight_modification = False

        return True


