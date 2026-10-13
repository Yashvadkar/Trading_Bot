# Superpowers Implementation Plan: Money-Aware Portfolio Capital Allocation, Small-Account Viability & Kronos/Championship Integration

## Goal
Implement mathematically rigorous portfolio cash awareness, Multi-Asset Covariance Kelly allocation using our existing Ledoit-Wolf shrunken covariance matrix ($\Sigma_{\text{LW}}^{-1} M$), Goverdhan "Gon" Gajjala's championship risk framework (+805% 2023 US Investing Champion), small-account cardinality constraints with fractional share execution, an Almgren-Chriss retail bypass gate, and architectural foundations for the `Kronos` K-line Foundation Model (`shiyu-coder/Kronos`).

---

## Assumptions
1. **Capital Conservation**: Total capital allocated across all orders in any scan cycle must strictly satisfy $\sum_{i=1}^K \text{Capital}_i \le \text{Available Cash} \times 0.95$.
2. **True Multi-Asset Covariance Kelly**: Asset allocation must account for cross-asset correlation using the Ledoit-Wolf shrunken covariance matrix $\Sigma_{\text{LW}}^{-1} M$ rather than heuristic univariate water-filling sums.
3. **Conformal Risk-Constrained Sizing (No Hyper-Kelly Ruin)**: Progressive exposure adjusts the fixed risk-per-trade budget ceiling $R \in [0.25\%, 0.75\%]$ and conformal uncertainty intervals, NEVER scaling Kelly fractions into the hyper-Kelly geometric decay regime ($f > f^*$).
4. **Small-Account Cardinality & Fractional Execution**: For accounts under $\$10,000$, cardinality is strictly capped at $K \in [1, 3]$ positions to eliminate micro-account spread drag and fee compounding, using Alpaca fractional shares (down to $0.001$ shares).
5. **Almgren-Chriss Retail Bypass**: For orders $< 0.01 \times \text{ADV}_{5\text{m}}$, bypass hyperbolic order slicing to avoid artificial spread crossings; route via direct marketable limit orders.
6. **Zero Regressions**: All 70 existing unit tests must remain 100% passing.

---

## Plan

### Step 1: Conformal & Progressive Risk Governor in `risk/governor.py`
- **Files**:
  - `risk/governor.py`
  - `tests/test_progressive_exposure.py`
- **Change**:
  - Create `ProgressiveExposureTracker` inside `risk/governor.py`:
    - Tracks recent closed trades ($N=10$) with fields: `pnl`, `r_multiple`, `is_win`.
    - Computes safe risk budget ceiling $R_{\text{eff}} = R_{\text{base}} \times \mathcal{M}_{\text{prog}}$:
      - **Contraction Mode**: If $\ge 2$ consecutive losses or rolling win rate $< 25\%$, $\mathcal{M}_{\text{prog}} = 0.50$ (slashes base risk from $0.50\%$ to $0.25\%$).
      - **Neutral Mode**: $\mathcal{M}_{\text{prog}} = 1.0$ (base risk $0.50\%$).
      - **Expansion Mode**: If $\ge 2$ consecutive wins with average win $\ge 2.0R$ and market in confirmed uptrend (SPY > 21 EMA), $\mathcal{M}_{\text{prog}} = 1.50$ (scales risk ceiling to $0.75\%$).
    - Prevents hyper-Kelly geometric ruin by capping single-position Kelly fraction at $\kappa_{\max} = 0.25$ (Quarter-Kelly).
  - Add unit tests verifying streak contraction, expansion limits, and risk ceiling enforcement.
- **Verify**:
  - `python -m unittest tests/test_progressive_exposure.py`

---

### Step 2: Multi-Asset Covariance Kelly & Cardinality-Constrained Allocator in `risk/governor.py`
- **Files**:
  - `risk/governor.py`
  - `tests/test_portfolio_allocator.py`
- **Change**:
  - Create `PortfolioCapitalAllocator` in `risk/governor.py`:
    - Method `allocate_batch_directives(candidates, portfolio_equity, available_cash, returns_matrix, progressive_mult)`:
      - Accepts list of candidate evaluation intents (ticker, price, ATR, stop_dist, debate_confidence).
      - **Small-Account Cardinality Constraint**:
        $$K = \min(\text{max\_concurrent}, \begin{cases} 2 & \text{if } \text{Equity} < \$5,000 \\ 3 & \text{if } \$5,000 \le \text{Equity} < \$10,000 \\ \text{max\_concurrent} & \text{otherwise} \end{cases})$$
      - Ranks candidates by conviction score $S_i = \text{confidence}_i \times \text{Sortino}_i$. Selects top $K$ candidates.
      - **Multi-Asset Covariance Kelly Vector**:
        Computes Ledoit-Wolf shrunken covariance matrix $\Sigma_{\text{LW}}$ across candidate returns.
        Solves $F^* = \kappa \cdot \Sigma_{\text{LW}}^{-1} M$, clipped to $[0.0, 0.25]$ per asset.
      - Normalizes weights across top $K$ candidates: $w_i = \frac{F_i^*}{\sum_{j=1}^K F_j^*}$.
      - Global Cash Pool: $C_{\text{pool}} = \text{available\_cash} \times 0.95$.
      - Individual Target Capital: $\text{Cap}_i = \min(w_i \cdot C_{\text{pool}}, \text{portfolio\_equity} \cdot 0.25)$.
      - Dual Constraint Position Sizing:
        $$\text{Shares}_i = \min\left(\frac{\text{Cap}_i}{P_i}, \frac{\text{Equity} \cdot R_{\text{eff}}}{\text{StopDist}_i}\right)$$
      - Strictly guarantees $\sum_{i=1}^K (\text{Shares}_i \cdot P_i) \le C_{\text{pool}}$ without FIFO starvation.
- **Verify**:
  - `python -m unittest tests/test_portfolio_allocator.py`

---

### Step 3: Almgren-Chriss Retail Execution Bypass & Fractional Share Routing
- **Files**:
  - `risk/governor.py`
  - `broker/alpaca_client.py`
  - `tests/test_small_account_sizing.py`
- **Change**:
  - In `risk/governor.py`:
    - In `AlmgrenChrissSlicer`: add volume threshold bypass:
      ```python
      if order_shares < 0.01 * (adv_5m or 100_000):
          # Retail-sized order: bypass slicing, single direct block to avoid spread drag
          return [order_shares]
      ```
    - Support fractional shares when `shares < 1.0` or `portfolio_equity < 10000.0`:
      `shares = round(shares, 3)` instead of `int(shares)`.
  - In `broker/alpaca_client.py`:
    - Format `qty=f"{directive.shares:.3f}"` in Alpaca REST payload when `shares` is a float.
  - Add unit tests verifying small accounts (\$1,000) can buy fractional quantities without rounding to 0.
- **Verify**:
  - `python -m unittest tests/test_small_account_sizing.py`

---

### Step 4: Deterministic Intraday VCP Pattern Detector in `screening/screener.py`
- **Files**:
  - `screening/screener.py`
  - `features/indicators.py`
  - `tests/test_vcp_screener.py`
- **Change**:
  - Implement programmatic Volatility Contraction Pattern (VCP) detection:
    - Peak & trough detection via progressive swing depth $D_1 > D_2 > D_3$.
    - Volume dry-up condition: final contraction volume $< 0.50 \times \text{SMA}(V, 50)$.
    - Intraday moving average perfect order: $\text{Price} > \text{EMA}(9) > \text{EMA}(21)$.
    - Expose `detect_intraday_vcp(bars_df) -> tuple[bool, float]` returning `(is_vcp, contraction_ratio)`.
- **Verify**:
  - `python -m unittest tests/test_vcp_screener.py`

---

### Step 5: Wire Two-Stage Batch Arena, Broker Feedback & Kronos Adapter in `bot.py`
- **Files**:
  - `bot.py`
  - `config.py`
  - `tests/test_pipeline.py`
- **Change**:
  - In `config.py`: Add `allow_fractional_shares: bool = True`, `max_cardinality_small_account: int = 2`, `base_risk_per_trade: float = 0.005`.
  - In `bot.py` (`TradingBot.run_cycle()`):
    - Replace the individual FIFO loop with a clean Two-Stage Batch Pipeline:
      1. **Debate Stage**: Run Neural Debate Arena across all screened stocks, collecting approved candidate signals.
      2. **Portfolio Allocation Stage**: Pass all approved candidates to `self.governor.allocator.allocate_batch_directives()` with multi-asset covariance matrix.
      3. **Execution Stage**: Iterate over allocated directives, route via retail bypass / Alpaca broker.
    - Wire trade outcome feedback from `TradeLedger` into `self.governor.progressive_tracker`.
    - Provide structural interface/stub for `KronosPredictor` foundation model embeddings (`model/kronos_adapter.py`).
- **Verify**:
  - `python -m unittest tests/test_pipeline.py`
  - `python -m unittest discover -s tests -p "test_*.py"`

---

### Step 6: Master Regression & Multi-Scale Autonomous Simulation
- **Files**:
  - Full test suite
  - `main.py`
- **Change**:
  - Run full discovery test suite: `python -m unittest discover -s tests -p "test_*.py"` (target $> 78$ tests passing).
  - Run dry-run cycle on simulated small account (\$1,000) and standard account (\$100,000):
    `python main.py --cycles 1 --dry-run`
  - Update `artifacts/superpowers/execution.md`, `review.md`, and `finish.md`.
- **Verify**:
  - Zero capital overdraw ($\sum \text{Allocated} \le 0.95 \times \text{Cash}$).
  - Small accounts successfully execute fractional positions with max 2 concurrent holdings.

---

## Risks & Mitigations
| Risk | Severity | Mitigation |
|---|---|---|
| Illiquid tickers rejecting fractional orders | Medium | Alpaca client checks `is_fractionable` or falls back to nearest valid share quantity. |
| Inverting ill-conditioned covariance matrix in multi-asset Kelly | High | Guaranteed well-conditioned inversion via already-integrated Ledoit-Wolf shrinkage ($\Sigma_{\text{LW}}^{-1}$). |
| Streak expansion causing drawdown in choppy market | High | Exposure expansion requires confirmed market index trend ($\text{SPY} > 21\text{ EMA}$) and is capped at Quarter-Kelly ($\kappa = 0.25$). |

---

## Rollback Plan
- All changes managed in Git.
- If regression occurs, `git restore` reverts to working commit `2ae8cd5`.
