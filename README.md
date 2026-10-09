# Hybrid Agent-Reach + Multi-Neural-Network Debate Trading Bot

Autonomous neuro-symbolic equity trading bot combining:
1. **Open-Ended Web Discovery** via `agent-reach` (Jina Reader, Exa Search, Financial RSS, Xueqiu, Reddit, Twitter) and high-speed text sanitization.
2. **Quantitative Liquidity Screening** (Market Cap $> \$500\text{M}$, ADV $> \$10\text{M}$, Price $> \$5$) via `yfinance`.
3. **Multi-Neural-Network Debate Architecture** (`BullNetwork`, `BearNetwork`, `ArbiterNetwork`) in PyTorch.
4. **Epistemic Disagreement Gating** ($\sigma^2_{debate}$) that scales down risk or forces `HOLD` during regime uncertainty.
5. **Deterministic Software 1.0 Risk Governor** (1.5% max risk per trade, 2x ATR trailing stop-loss, 3% daily drawdown circuit breaker).
6. **Alpaca Paper Trading REST Gateway** with persistent SQLite audit telemetry (`trades.db`).

---

## 🏛️ System Architecture

```text
       ┌─────────────────────────────────────────────────────────────┐
       │                1. OPEN-ENDED WEB DISCOVERY                  │
       │   agent-reach (Jina Reader, Exa, RSS, Xueqiu, Reddit, FinTwit│
       └──────────────────────────────┬──────────────────────────────┘
                                      │ Raw Articles & Headlines
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             2. TEXT SANITIZER & TICKER EXTRACTOR            │
       │   - Regex / NER Symbol Extraction                           │
       │   - Prunes 70+ False Positive Stopwords ($AI, $ALL, $FOR)   │
       │   - Financial Lexicon Sentiment & Catalyst Scoring          │
       └──────────────────────────────┬──────────────────────────────┘
                                      │ Extracted Ticker Candidates
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             3. QUANTITATIVE LIQUIDITY SCREENER              │
       │   - ADV > $10M | Market Cap > $500M | Price > $5.00         │
       │   - Vectorized Technicals: RSI, MACD, ATR, SMA Ratios, BB%B │
       └──────────────────────────────┬──────────────────────────────┘
                                      │ 32-dim Normalized State Vector [s_t]
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │         4. MULTI-NEURAL-NETWORK DEBATE ENSEMBLE             │
       │   ┌───────────────────────────┐ ┌─────────────────────────┐ │
       │   │  Bull Network (GRU-128)   │ │  Bear Network (GRU-128) │ │
       │   │  Optimistic Catalyst Peak │ │  Adversarial Risk Seeker│ │
       │   └─────────────┬─────────────┘ └────────────┬────────────┘ │
       │                 │ [h_bull, P_bull]           │ [h_bear, P_bear]
       │                 └─────────────┬──────────────┘              │
       │                               ▼                             │
       │                 ┌──────────────────────────┐                │
       │                 │ Arbiter Network (MLP-64) │                │
       │                 │ Measures Disagreement σ² │                │
       │                 └─────────────┬────────────┘                │
       └───────────────────────────────┼─────────────────────────────┘
                                       │ Action Proposal & Confidence
                                       ▼
       ┌─────────────────────────────────────────────────────────────┐
       │         5. DETERMINISTIC RISK GOVERNOR & SAFETY GATE        │
       │   - Shares = (Portfolio Equity * 1.5%) / (2x ATR)           │
       │   - Max 20% Capital Allocation per Position                 │
       │   - Circuit Breaker: 3% daily drawdown halt / 5% liquidate  │
       └──────────────────────────────┬──────────────────────────────┘
                                       │ Authorized Order Directive
                                       ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                 6. BROKER GATEWAY & LEDGER                  │
       │   - Alpaca Paper REST API (Bracket Orders with Trailing Stop│
       │   - SQLite Audit Ledger (trades.db)                         │
       │   - Replay Buffer Online Co-Training Update                 │
       └─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Requirements & Environment
The bot requires Python 3.10+ and standard packages (PyTorch, NumPy, yfinance, requests):
```bash
# Verify credentials in .env
cat .env
```

### 2. Check Account Telemetry & Positions
```bash
python main.py --status
```

### 3. Run Internet Discovery Scan Only
```bash
python main.py --discover
```

### 4. Run a Simulated Dry-Run Cycle (No real orders placed)
```bash
python main.py --dry-run
```

### 5. Run 24/7 Autonomous Daemon Mode (Recommended)
Runs an infinite continuous loop with automatic US market clock awareness, active position management, and off-hours neural training:
```bash
python main.py --daemon
# Optional: customize scan interval (e.g. 60 seconds)
python main.py --daemon --interval 60
```

### 6. Run Fixed-Cycle Paper Trading
```bash
# Run 5 consecutive cycles with 60s scan interval
python main.py --mode paper --cycles 5 --interval 60
```

### 7. Run Unit & Regression Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🛡️ Risk Management Invariants

1. **Software 1.0 Air-Gap**: The neural networks only propose directional conviction; deterministic Python logic computes the exact mathematical share sizing and stop-loss price. Under no circumstances can a neural network bypass the 1.5% max risk threshold.
2. **Epistemic Uncertainty Gate**: When Bull and Bear networks violently disagree ($\sigma^2_{debate} > 0.35$), the Arbiter automatically defaults to action 0 (`HOLD`), preserving capital during noisy market regimes.
3. **Daily Drawdown Circuit Breaker**:
   - $\ge 3.0\%$ Loss from start-of-day equity: New buy orders are strictly halted.
   - $\ge 5.0\%$ Loss: Emergency mode triggers full position exit.
4. **Secret Isolation**: Alpaca keys are stored strictly in `.env`, which is permanently `.gitignore`d and never logged in plain text.

---

## 📊 File Structure

```text
Trading_Bot/
├── .env                  # Isolated Alpaca credentials & trading hyperparameters
├── .gitignore            # Git exclusion rules for .env, *.db, and *.pt weights
├── config.py             # Typed dataclass config loader with credential masking
├── bot.py                # Master autonomous orchestrator loop
├── main.py               # CLI entrypoint with --mode, --status, --dry-run flags
├── discovery/
│   ├── cleaner.py        # Financial text sanitizer & false-positive ticker filter
│   └── scout.py          # Open-ended discovery bridge (RSS, Jina Reader)
├── screening/
│   └── screener.py       # Quantitative liquidity & health screening (yfinance)
├── features/
│   ├── indicators.py     # Deterministic vectorized technical indicators (RSI, MACD, ATR)
│   └── normalizer.py     # 32-dimensional rolling state vector normalizer
├── model/
│   ├── networks.py       # PyTorch Bull, Bear, and Arbiter neural network modules
│   ├── debate.py         # Adversarial debate arena & disagreement variance
│   ├── replay_buffer.py  # Prioritized experience replay buffer
│   ├── trainer.py        # Asymmetric ensemble co-trainer
│   └── pretrain.py       # Warm-start pretraining utility
├── risk/
│   ├── governor.py       # Software 1.0 ATR position sizer and risk firewall
│   └── circuit_breaker.py# Portfolio drawdown monitor and emergency halt
├── broker/
│   ├── alpaca_client.py  # Alpaca Paper REST API client (bracket orders)
│   └── ledger.py         # SQLite audit and telemetry database (trades.db)
└── tests/
    └── test_pipeline.py  # Full unit and regression test suite
```
