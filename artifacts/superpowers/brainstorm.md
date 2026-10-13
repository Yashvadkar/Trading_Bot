# Superpowers Brainstorm: Money-Aware Portfolio Capital Allocation & Championship Risk Integration

## Goal
Transform `Trading_Bot` from an isolated, single-order sizer into a **globally money-aware, small-account viable portfolio trading engine**. Integrate:
1. **Simultaneous Multi-Asset Portfolio Kelly Allocation & Cycle Cash Budgeting**: Ensure multiple opportunities in a single cycle divide available cash strictly without exceeding buying power or starving smaller accounts.
2. **Goverdhan "Gon" Gajjala (2023 US Investing Champion, +805% Return) Risk Architecture**:
   - Asymmetric risk/reward modeling (targeting 2.5:1 to 3.5:1 payoff with 0.25%–0.50% account risk per trade).
   - Minervini/Gajjala Progressive Exposure engine (expanding position limits and risk during win streaks; drastically throttling to cash during drawdowns).
   - Intraday Volatility Contraction Pattern (VCP) and 9/21 EMA trend anchor screening.
   - Partial scaling (selling 50% at 2R–3R, moving stop to breakeven, trailing runners on 9 EMA).
3. **Small-Account Real-Money Adaptability**: Support fractional share execution via Alpaca `notional` or fractional quantities, preventing share rounding to 0 for accounts with $500–$5,000 capital.

---

## Constraints
1. **Capital Conservation (No Buying Power Overdraw)**: Total authorized capital in any scanning cycle must strictly satisfy $\sum_{i} \text{Capital}_i \le \text{Available Cash} \times 0.95$.
2. **Deterministic Risk Budgeting**: Individual trade risk must never exceed the active risk-per-trade budget ($0.25\%$ to $1.5\%$ of total equity), calculated strictly as $\text{Shares} \times (\text{Entry} - \text{Stop})$.
3. **Broker Compatibility**: Maintain Alpaca Paper and Live REST/WebSocket execution idempotency while adding optional fractional share / notional execution for small accounts.
4. **Execution Latency**: Portfolio optimization across candidates must be instantaneous ($<5\text{ms}$) using closed-form analytical equations (Vectorized Fractional Kelly / Water-Filling budgeter) rather than heavy quadratic programming solvers that stall the event loop.
5. **No Regressions**: All 70 existing unit tests must continue passing without breaking.

---

## Known Context
- **Current Architecture Flaw in `bot.py`**:
  In `TradingBot.run_cycle()`, the loop over `screened_stocks` evaluates each stock against `account.cash` independently. If 8 stocks pass, each is evaluated as if 100% of cash is uncommitted, causing total authorized capital to exceed 139% of equity ($139k authorized on a $100k account).
- **Disconnected Kelly Inputs**:
  `risk/governor.py` possesses `calculate_fractional_kelly_capital()`, but `bot.py` never passes `returns_history` or `conformal_residuals` into `evaluate_order()`, causing the engine to fall back to a naive `min(0.20 * Equity, 0.95 * Cash)`.
- **Championship Research (Gon Gajala, 805% Return, TraderLion Video `10pHBNVi4Jc`)**:
  - Gajjala won the 2023 US Investing Championship stock division with an audited **+805% return** with only a **28%–31% win rate**.
  - Key mathematics: Average win was 8.5%–13% vs average loss capped at 4% (asymmetry > 2.5R).
  - Risk per trade was strictly capped at **0.25%–0.50%** of account equity.
  - Employed Minervini's **Progressive Exposure**: Starting small; after 2–3 consecutive winners, increase exposure; after 2 consecutive losers, halve risk and contract back to base size.
  - Core setups: Intraday Volatility Contraction Pattern (VCP) with volume drying up into tight consolidation pivots above 9 EMA & 21 EMA.

---

## Risks
1. **Small-Account Lumpy Sizing Without Fractional Shares**: On an account with $1,000, high-priced momentum stocks ($200–$1,000/share) cannot be purchased without fractional shares, or 1 share represents 50%+ of equity.
   *Mitigation*: Implement Alpaca fractional share order routing (or notional dollar orders) when account equity $< \$10,000$ or share price exceeds 10% of equity.
2. **Over-Diversification vs Under-Diversification**: Dividing a tiny account ($1,000) among 8 stocks leaves tiny $125 positions eaten up by bid-ask spread and slippage.
   *Mitigation*: Implement an optimal candidate selector that picks the top $K$ ($K = \min(\text{max\_concurrent}, \lfloor \text{Equity} / 500 \rfloor)$) highest-conviction opportunities using Debate confidence $\times$ Kelly fraction score.
3. **Premature Exposure Expansion in Volatile Regimes**: Expanding exposure too fast after temporary lucky wins could amplify subsequent drawdown.
   *Mitigation*: Require both positive rolling trade expectancy ($\ge 1.5R$) and market regime filter (SPY/QQQ above 21 EMA) before increasing progressive exposure above base 0.5%.

---

## Options (2–4)

### Option A: Local Cycle Cash Tracker (Minimal Patch)
- Simply decrement a mutable `cycle_available_cash` in `bot.py` as orders are authorized.
- *Pros*: Very easy to implement.
- *Cons*: First-come, first-served (FIFO) bias. The first ticker screened gets full capital; subsequent higher-conviction candidates get starved or rejected. Does not implement championship sizing or progressive exposure.

### Option B: Two-Stage Simultaneous Portfolio Kelly & Progressive Exposure Engine (Recommended)
- **Stage 1 (Batch Evaluation & Ranking)**: In each cycle, evaluate all approved debate candidates simultaneously. Compute each candidate's Conformal Fractional Kelly fraction $f_i^*$.
- **Stage 2 (Global Budget Allocation & Water-Filling)**:
  - Apply Gon Gajala's **Progressive Exposure Multiplier** $\mathcal{M}_{\text{prog}} \in [0.5, 2.0]$ based on rolling PnL streak and market trend.
  - Budget available cash across top $K$ candidates proportional to their normalized Kelly conviction:
    $$w_i = \frac{f_i^*}{\sum_{j=1}^K f_j^*}, \quad \text{Capital}_i = \min(w_i \cdot \text{Cash}_{\text{avail}}, \text{MaxCap}_i)$$
  - Size shares using strict $R$-budget: $\text{Shares}_i = \min\left(\lfloor \frac{\text{Capital}_i}{P_i} \rfloor, \lfloor \frac{\text{Equity} \cdot R_i}{\text{StopDist}_i} \rfloor\right)$.
  - Enable fractional shares if configured or if account equity $< \$10,000$.
- *Pros*: Mathematically sound, money-aware, prevents overdraw, scales from $500 to $1,000,000 accounts, implements championship asymmetric risk.
- *Cons*: Requires coordinating batch allocation in `RiskGovernor` before firing execution directives.

### Option C: Continuous Mean-Variance Quadratic Optimizer (Markowitz MVO)
- Solve full numerical quadratic programming (SLSQP / CVXPY) across portfolio covariance matrix.
- *Pros*: Theoretically optimal for static long-term portfolios.
- *Cons*: Overkill for swing/momentum day trading, sensitive to covariance estimation noise, adds heavy external dependencies, violates Ponytail anti-overengineering standard.

---

## Recommendation
**Adopt Option B: Two-Stage Simultaneous Portfolio Kelly & Progressive Exposure Engine.**
1. Implement `PortfolioCapitalAllocator` in `risk/governor.py` that takes all candidate directives in a cycle, computes global money budget, scales by Progressive Exposure, and allocates cash without FIFO bias.
2. Feed historical return series and conformal residuals from `historical_loader.py` into the Kelly engine.
3. Add Gajjala/Minervini Progressive Exposure tracking in `risk/governor.py` based on recent trade win/loss streaks recorded in `TradeLedger`.
4. Add fractional share support in `broker/alpaca_client.py` for small-account capital efficiency.

---

## Acceptance Criteria
1. **Cycle Money Awareness**: Total authorized capital across all orders in any single cycle never exceeds $95\%$ of available cash, verified in test suite.
2. **Conviction-Weighted Capital Distribution**: When multiple candidates qualify, capital is allocated proportional to their Kelly conviction score rather than FIFO starvation.
3. **Progressive Exposure Governance**: Risk per trade dynamically adjusts between $0.25\%$ (during drawdowns/losing streaks) and $1.0\%$ (during verified winning streaks).
4. **Small-Account Viability**: Verified on simulated $\$1,000$ and $\$5,000$ accounts without share rounding errors or excessive concentration.
5. **Full Regression Integrity**: All existing 70 tests pass + new portfolio allocation & progressive exposure test suites pass.
