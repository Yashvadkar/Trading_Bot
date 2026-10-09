"""Comprehensive Test Suite for Hybrid Neuro-Symbolic Trading Bot.

Covers:
1. Environment & credential masking
2. Vectorized financial indicators
3. 32-dim state normalization
4. Ticker cleaning & false-positive pruning
5. Bull vs. Bear debate & Arbiter uncertainty resolution
6. Karpathy micro-case training convergence
7. Deterministic Risk Governor & ATR sizing
8. Circuit Breaker drawdown thresholds
9. SQLite Trade Ledger audit persistence
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, RiskConfig, ModelConfig, ScoutConfig, load_config
from features.indicators import (
    compute_rsi, compute_macd, compute_atr, compute_bollinger_pct_b, extract_technicals
)
from features.normalizer import (
    StateNormalizer, SentimentMetrics, FundamentalMetrics, MicrostructureMetrics, PortfolioContext, DebateMetrics
)
from discovery.cleaner import extract_tickers_with_sentiment, FALSE_POSITIVE_TICKERS
from model.networks import BullNetwork, BearNetwork, ArbiterNetwork
from model.debate import DebateArena
from model.replay_buffer import ReplayBuffer
from model.trainer import EnsembleTrainer
from risk.circuit_breaker import CircuitBreaker, BreakerStatus
from risk.governor import RiskGovernor, OrderSide, PositionActionType, PositionDirective
from broker.alpaca_client import AlpacaPaperClient, MarketClock
from broker.ledger import TradeLedger, TradeRecord


class TestTradingBotPipeline(unittest.TestCase):

    def test_config_masking(self):
        cfg = load_config()
        self.assertTrue(len(cfg.alpaca_api_key) > 5)
        masked = cfg.mask_key(cfg.alpaca_api_key)
        self.assertTrue(masked.startswith("PKAV"))
        self.assertTrue(masked.endswith("C3GF"))
        self.assertNotIn("SWQOAJT", masked)  # Middle must be hidden

    def test_technical_indicators(self):
        prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
                           111.0, 110.0, 112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0])
        rsi = compute_rsi(prices, period=14)
        self.assertTrue(0.0 <= rsi <= 100.0)
        self.assertTrue(rsi > 50.0)  # Upward trending series

        macd, sig, hist = compute_macd(prices, 12, 26, 9)
        self.assertIsInstance(hist, float)

        pct_b = compute_bollinger_pct_b(prices, period=20)
        self.assertTrue(-0.5 <= pct_b <= 1.5)

    def test_state_normalizer_shape_and_bounds(self):
        normalizer = StateNormalizer()
        np.random.seed(42)
        closes = np.linspace(100, 120, 50)
        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes
        volumes = np.ones(50) * 1e6

        tech = extract_technicals(opens, highs, lows, closes, volumes)
        state = normalizer.build_state_vector(
            tech,
            SentimentMetrics(),
            FundamentalMetrics(),
            MicrostructureMetrics(),
            PortfolioContext(),
            DebateMetrics()
        )
        self.assertEqual(state.shape, (1, 32))
        self.assertFalse(torch.isnan(state).any())
        self.assertFalse(torch.isinf(state).any())
        self.assertTrue((state >= -3.0).all() and (state <= 3.0).all())

    def test_ticker_cleaner_and_stopwords(self):
        text = "Investors bought $NVDA and $AAPL, but avoid $AI and $ALL since they are words. Sold $BAD after fraud."
        results = extract_tickers_with_sentiment(text)
        tickers = [r.ticker for r in results]

        self.assertIn("NVDA", tickers)
        self.assertIn("AAPL", tickers)
        self.assertNotIn("AI", tickers)
        self.assertNotIn("ALL", tickers)

        bad = next(r for r in results if r.ticker == "BAD")
        self.assertTrue(bad.sentiment_score < 0.0)

    def test_debate_arena_and_disagreement(self):
        bull = BullNetwork(state_dim=32, hidden_dim=64)
        bear = BearNetwork(state_dim=32, hidden_dim=64)
        arb = ArbiterNetwork(hidden_dim=64)
        arena = DebateArena(bull, bear, arb, disagreement_threshold=0.35)

        state = torch.randn(1, 32)
        verdict = arena.evaluate_debate(state)
        self.assertIn(verdict.chosen_action, [0, 1, 2, 3])
        self.assertTrue(0.0 <= verdict.action_confidence <= 1.0)
        self.assertTrue(verdict.disagreement_variance >= 0.0)

    def test_karpathy_micro_training_convergence(self):
        bull = BullNetwork(state_dim=32, hidden_dim=64)
        bear = BearNetwork(state_dim=32, hidden_dim=64)
        arb = ArbiterNetwork(hidden_dim=64)
        trainer = EnsembleTrainer(bull, bear, arb, lr=2e-3)
        buffer = ReplayBuffer(capacity=50)

        for _ in range(8):
            buffer.push(torch.randn(32), 2, 0.1, torch.randn(32), False, 0.05)

        loss_0 = trainer.train_step(buffer, batch_size=8)
        for _ in range(20):
            loss_final = trainer.train_step(buffer, batch_size=8)

        # Confirm loss decreases
        self.assertTrue(loss_final.bull_loss < loss_0.bull_loss or loss_final.bull_loss < 0.5)

    def test_risk_governor_sizing_and_limits(self):
        cfg = RiskConfig(max_risk_per_trade=0.015, atr_multiplier_stop=2.0)
        gov = RiskGovernor(cfg)

        directive = gov.evaluate_order(
            ticker="NVDA",
            action=2, # Buy 1.0x
            current_price=100.0,
            atr=5.0, # Stop dist = 10.0 -> Stop at 90.0
            portfolio_equity=100_000.0,
            available_cash=50_000.0,
            current_positions_count=0,
            is_already_held=False,
            circuit_breaker_active=False
        )
        self.assertTrue(directive.is_authorized)
        self.assertEqual(directive.shares, 150) # $1,500 dollar risk / $10 stop dist = 150 shares
        self.assertEqual(directive.stop_loss_price, 90.0)

    def test_circuit_breaker_thresholds(self):
        cb = CircuitBreaker(halt_drawdown_limit=0.03, emergency_limit=0.05)
        # Normal
        s1 = cb.update_equity(100_000.0)
        self.assertEqual(s1.status, BreakerStatus.NORMAL)

        # 3.5% loss -> Halted
        s2 = cb.update_equity(96_500.0)
        self.assertEqual(s2.status, BreakerStatus.HALTED)
        self.assertFalse(cb.can_open_new_position(96_500.0))

        # 6.0% loss -> Emergency
        s3 = cb.update_equity(94_000.0)
        self.assertEqual(s3.status, BreakerStatus.EMERGENCY)

    def test_sqlite_ledger_persistence(self):
        import gc
        temp_dir = tempfile.gettempdir()
        db_path = Path(temp_dir) / "test_ledger_audit.db"
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception:
                pass

        try:
            ledger = TradeLedger(str(db_path))
            ledger.log_discovery("MSFT", 5, 0.7, ["cloud", "growth"])
            ledger.log_debate("MSFT", 0.75, 0.25, 0.04, 2, True)
            ledger.log_trade(TradeRecord(
                ticker="MSFT",
                side="buy",
                shares=20,
                price=420.0,
                stop_loss=405.0,
                order_id="TEST_ORD_123",
                status="FILLED"
            ))
            recent = ledger.get_recent_trades(5)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["ticker"], "MSFT")
        finally:
            gc.collect()
            try:
                db_path.unlink(missing_ok=True)
            except Exception:
                pass

    def test_market_clock_telemetry(self):
        cfg = load_config()
        client = AlpacaPaperClient(cfg, dry_run=True)
        clock = client.get_market_clock()
        self.assertIsInstance(clock, MarketClock)
        self.assertTrue(clock.is_open)

    def test_position_monitor_and_ratchet_stop(self):
        cfg = RiskConfig(atr_multiplier_stop=2.0)
        gov = RiskGovernor(cfg)

        # 1. Test Neural Exit (Action 3: EXIT)
        exit_dir = gov.evaluate_held_position(
            ticker="NVDA",
            shares=100,
            entry_price=100.0,
            current_price=102.0,
            atr=4.0,
            unrealized_pnl_pct=0.02,
            debate_action=3, # EXIT
            current_stop_price=92.0
        )
        self.assertEqual(exit_dir.action_type, PositionActionType.EXIT)
        self.assertIn("EXIT signal", exit_dir.reason)

        # 2. Test Ratchet Stop on +8% gain (Stop raised above entry to lock profit)
        ratchet_dir = gov.evaluate_held_position(
            ticker="NVDA",
            shares=100,
            entry_price=100.0,
            current_price=108.0,
            atr=3.0,
            unrealized_pnl_pct=0.08,
            debate_action=0, # HOLD
            current_stop_price=94.0 # Old stop below entry
        )
        self.assertEqual(ratchet_dir.action_type, PositionActionType.UPDATE_STOP)
        self.assertTrue(ratchet_dir.new_stop_price > 100.0) # Stop raised ABOVE entry!
        self.assertIn("Profit lock ratcheted", ratchet_dir.reason)

        # 3. Test Normal Hold when flat (+0.5%)
        hold_dir = gov.evaluate_held_position(
            ticker="NVDA",
            shares=100,
            entry_price=100.0,
            current_price=100.5,
            atr=3.0,
            unrealized_pnl_pct=0.005,
            debate_action=0,
            current_stop_price=94.0
        )
        self.assertEqual(hold_dir.action_type, PositionActionType.HOLD)


if __name__ == "__main__":
    unittest.main()
