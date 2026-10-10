# Superpowers Implementation Plan — 4th Institutional Audit Remediation

## Goal
Remediate all vulnerabilities and architectural blockers identified in the 4th Deep Research Institutional Audit of `Trading_Bot` (advancing Production-Readiness from 7.5/10 to 9.8/10):
1. **Blocker: Asynchronous Persistence** — Eliminate WAL checkpoint starvation by removing PASSIVE checkpoints from the synchronous write path and introducing a dedicated background daemon executing periodic `PRAGMA wal_checkpoint(RESTART);` with clean process lifecycle management.
2. **Major: Monotonic SSE Sequence Gating** — Implement ULID lexicographical sorting and Last-Write-Wins (CRDT) sequence gating in `OrderLifecycleFSM` to prevent non-deterministic, out-of-order thread pool Alpaca Activity SSE events from overwriting terminal states.
3. **Major: Survivorship Bias Elimination** — Enhance Point-in-Time (PIT) filtration by mapping both `CONSTITUENT_INCEPTION` and `CONSTITUENT_DELISTING` to evaluate the boolean window $t \in [t_{inception}, t_{delisting})$, mathematically preventing historical lookahead and survivorship contamination.
4. **Minor: Microstructural Execution** — Dynamically recalculate Almgren-Chriss remaining urgency parameter $\kappa$ and temporary impact penalty $\tilde{\eta}$ whenever Cont-Larrard queue fill probability ($P_{\text{fill}} < 0.40$) triggers an aggressive market sweep.
5. **Nit: Spectral Stability** — Implement analytical dynamic Ledoit-Wolf shrinkage based on the Frobenius norm minimization to optimally bound the correlation condition number during market stress regimes.
6. **Master Verification**: Expand the regression suite ($\ge 64$ tests) and execute live autonomous dry-run cycle without warnings or regressions.

---

## Assumptions
1. **Target Directory**: All modifications reside within `c:/Users/YASH/AIGaming/Trading_Bot/`.
2. **Baseline Stability**: All 59 existing unit tests must continue to pass cleanly without regression.
3. **Ponytail Standard**: Zero unneeded external dependencies; solutions leverage native Python `threading`, standard library SQLite, and memory-contiguous NumPy structures.
4. **Broker Protocol**: Full compliance with Alpaca v2 REST and WebSocket Activity Streams (`v2beta1/events/activities`).

---

## Plan

### Step 1: Dedicated WAL Checkpoint Daemon & Starvation Elimination
- **Files**:
  - `Trading_Bot/broker/ledger.py`
  - `Trading_Bot/tests/test_ledger_wal.py`
- **Change**:
  - In `broker/ledger.py`:
    - Remove `PRAGMA wal_checkpoint(PASSIVE);` from the latency-sensitive write path inside `_sync_batch_write`.
    - Implement a dedicated background daemon `_start_checkpoint_daemon(interval_sec=300)` running a dedicated thread that executes `PRAGMA wal_checkpoint(RESTART);` periodically.
    - Implement `stop_checkpoint_daemon()` using a `threading.Event` to ensure clean shutdown without thread leaks.
    - Ensure `checkpoint_wal(mode="RESTART")` can be called explicitly at trading cycle conclusion or application exit.
- **Verify**:
  - `python -m unittest tests/test_ledger_wal.py` verifying daemon checkpointing, clean thread termination, and RESTART truncation.

### Step 2: Monotonic ULID-Based SSE Sequence Gating in Order Lifecycle FSM
- **Files**:
  - `Trading_Bot/broker/order_fsm.py`
  - `Trading_Bot/broker/alpaca_client.py`
  - `Trading_Bot/tests/test_order_fsm.py`
- **Change**:
  - In `broker/order_fsm.py`:
    - Add `last_event_id: str | None = None` to `OrderStateRecord`.
    - Implement `process_sse_event(client_order_id: str, new_state: OrderState, event_id: str, filled_qty: int = 0) -> bool`:
      - Validates lexicographical ULID ordering: if `event_id <= record.last_event_id`, discards stale/duplicate events.
      - Enforces terminal immutability: if record is already in a terminal state (`FILLED`, `CANCELED`, `REJECTED`), discards any delayed active state transitions (e.g. out-of-order `PARTIALLY_FILLED`).
      - Updates `record.last_event_id = event_id` monotonically.
  - In `broker/alpaca_client.py`:
    - Wire WebSocket/SSE event handlers to pass the incoming event ULID through `process_sse_event()`.
- **Verify**:
  - `python -m unittest tests/test_order_fsm.py` asserting that delayed SSE messages with older event IDs or terminal reversions are rejected.

### Step 3: Survivorship Bias Mitigation via Delisting Filtration
- **Files**:
  - `Trading_Bot/data/historical_loader.py`
  - `Trading_Bot/tests/test_pit_filtration.py`
- **Change**:
  - In `data/historical_loader.py`:
    - Add `CONSTITUENT_DELISTING: dict[str, str | None]` mapping constituent termination dates alongside `CONSTITUENT_INCEPTION`.
    - Include representative historical delisted assets (e.g., `TWTR`: "2022-10-27", `BBBY`: "2023-05-03", `LEHM`: "2008-09-15").
    - Implement `is_constituent_tradable(ticker: str, as_of_date: str) -> bool` asserting $t \in [t_{\text{inception}}, t_{\text{delisting}})$.
    - Update `build_historical_replay_buffer()` and screener to enforce that delisted tickers cannot generate signals or execute trades after their delisting date, eliminating Class I survivorship contamination.
- **Verify**:
  - `python -m unittest tests/test_pit_filtration.py` asserting that queries evaluated after delisting dates return `False` and reject state vector generation.

### Step 4: Almgren-Chriss Dynamic Urgency Recalculation on Aggressive Sweeps
- **Files**:
  - `Trading_Bot/risk/governor.py`
  - `Trading_Bot/tests/test_institutional_risk.py`
- **Change**:
  - In `risk/governor.py`:
    - In `AlmgrenChrissSlicer`:
      - Implement `recalculate_urgency_on_spread_crossing(current_urgency: float, spread_penalty: float = 0.25) -> float`:
        $$\tilde{\eta}_{\text{adj}} = \tilde{\eta} \cdot (1 + \text{spread\_penalty}), \quad \kappa_{\text{adj}} = \sqrt{\frac{\lambda \sigma^2}{\tilde{\eta}_{\text{adj}}}}$$
      - When Cont-Larrard queue probability ($P_{\text{fill}} < 0.40$) triggers an `AGGRESSIVE_MARKET` sweep, dynamically adjust the remaining inventory urgency $\kappa$ to prevent shortfall variance runaway.
      - Update `route_child_slices()` to output the recalibrated urgency for subsequent slices.
- **Verify**:
  - `python -m unittest tests/test_institutional_risk.py` asserting that aggressive sweeps trigger urgency damping for remaining child orders.

### Step 5: Dynamic Frobenius-Norm Ledoit-Wolf Linear Shrinkage
- **Files**:
  - `Trading_Bot/features/quant_theories.py`
  - `Trading_Bot/tests/test_quant_theories.py`
- **Change**:
  - In `features/quant_theories.py`:
    - In `QuantTheoryEngine.clean_correlation_rmt()`:
      - Add dynamic Ledoit-Wolf intensity calculation:
        $$\alpha^* = \text{clip}\left(\frac{\sum_{i \ne j} \text{Var}(r_{ij})}{\|C_{\text{clean}} - I\|_F^2}, 0.01, 0.40\right)$$
      - Apply dynamic shrinkage when `dynamic_shrinkage=True`, using static $\alpha = 0.05$ as a calibrated baseline.
      - Ensures the condition number is optimally minimized during severe market crises and volatility shocks.
- **Verify**:
  - `python -m unittest tests/test_quant_theories.py` asserting that dynamic shrinkage minimizes condition number across synthetic high-stress market matrices.

### Step 6: Master Bot Integration, Full Discovery & Dry-Run Cycle
- **Files**:
  - `Trading_Bot/bot.py`
  - `Trading_Bot/main.py`
- **Change**:
  - In `TradingBot.__init__`: start the ledger WAL checkpoint daemon on startup and register graceful daemon shutdown on cycle/application exit.
  - Wire ULID sequence gating into broker event handlers and integrate dynamic urgency tracking into child order routing.
  - Execute full test discovery across all test modules (target: $\ge 64$ passing tests).
  - Execute autonomous live dry-run cycle verifying clean execution and zero warnings.
- **Verify**:
  - `python -m unittest discover -s tests -p "test_*.py"`
  - `python main.py --cycles 1 --dry-run`

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Checkpoint Thread Blocking** | RESTART checkpoint hangs if a rogue reader holds an open cursor. | Bound the connection timeout to 10.0s and catch `sqlite3.OperationalError` safely; continue next cycle. |
| **ULID Sorting Edge Cases** | Malformed or empty event IDs could silently reject valid updates. | Fallback to timestamp comparison if event ID is missing or not a valid ULID. |
| **Delisting Lookahead False Positives** | Assets incorrectly excluded if delisting timestamp timezone does not match UTC. | Standardize all inception and delisting timestamps strictly to ISO-8601 UTC. |
| **Dynamic Shrinkage Denominator Zero** | $\|C - I\|_F^2$ near zero if correlation matrix is already identity. | Enforce minimum epsilon $10^{-8}$ in denominator and clip $\alpha^*$ to $[0.01, 0.40]$. |

---

## Rollback Plan
If any step introduces instability:
1. Revert specific modified module via `git checkout -- <file>`.
2. Existing 59 unit tests serve as the baseline; verify with `python -m unittest discover -s tests -p "test_*.py"`.
