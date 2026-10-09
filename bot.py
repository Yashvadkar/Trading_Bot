"""Master Autonomous Trading Bot Orchestrator.

Glues together:
1. Open-Ended Scout Discovery (agent-reach / RSS / Jina)
2. Quantitative Liquidity & Fundamental Screener (yfinance)
3. 32-dim State Normalization Engine
4. Multi-Neural-Network Debate Arena (Bull vs. Bear vs. Arbiter)
5. Deterministic Risk Governor & Drawdown Circuit Breaker
6. Alpaca Paper Broker Gateway & Persistent SQLite Ledger
7. Asymmetric Co-Training Replay Buffer Loop
"""

from __future__ import annotations
import time
from pathlib import Path
from dataclasses import dataclass
import torch
from config import Config
from discovery.scout import DiscoveryScout
from screening.screener import QuantitativeScreener
from features.normalizer import StateNormalizer, SentimentMetrics, PortfolioContext, DebateMetrics
from model.networks import BullNetwork, BearNetwork, ArbiterNetwork
from model.debate import DebateArena, DebateVerdict
from model.replay_buffer import ReplayBuffer
from model.trainer import EnsembleTrainer
from model.pretrain import run_pretraining
from risk.circuit_breaker import CircuitBreaker, BreakerStatus
from risk.governor import RiskGovernor, OrderDirective, OrderSide, PositionDirective, PositionActionType
from broker.alpaca_client import AlpacaPaperClient, AccountStatus, PositionInfo
from broker.ledger import TradeLedger, TradeRecord


@dataclass
class CycleSummary:
    timestamp: float
    discovered_count: int
    screened_count: int
    orders_authorized: int
    orders_executed: int
    circuit_breaker_status: str
    total_equity: float


class TradingBot:
    """Master neuro-symbolic algorithmic trading engine."""

    def __init__(self, config: Config, dry_run: bool = False, db_path: str = "trades.db"):
        self.config = config
        self.dry_run = dry_run

        # Core Components
        self.scout = DiscoveryScout()
        self.screener = QuantitativeScreener(config.risk)
        self.normalizer = StateNormalizer()

        # Neural Debate Ensemble
        self.bull_net = BullNetwork(state_dim=config.model.state_dim, hidden_dim=config.model.hidden_dim)
        self.bear_net = BearNetwork(state_dim=config.model.state_dim, hidden_dim=config.model.hidden_dim)
        self.arbiter_net = ArbiterNetwork(hidden_dim=config.model.hidden_dim)
        self.arena = DebateArena(
            self.bull_net,
            self.bear_net,
            self.arbiter_net,
            disagreement_threshold=config.model.disagreement_threshold
        )

        # Experience Replay & Co-Trainer
        self.buffer = ReplayBuffer(capacity=config.model.replay_capacity)
        self.trainer = EnsembleTrainer(self.bull_net, self.bear_net, self.arbiter_net, lr=config.model.learning_rate)

        # Risk Software 1.0 Firewalls
        self.governor = RiskGovernor(config.risk)
        self.circuit_breaker = CircuitBreaker(
            halt_drawdown_limit=config.risk.daily_drawdown_limit,
            emergency_limit=config.risk.emergency_liquidation
        )

        # Broker & Audit Ledger
        self.broker = AlpacaPaperClient(config, dry_run=dry_run)
        self.ledger = TradeLedger(db_path)

        # Warm start neural network weights from offline pre-trained checkpoint
        weights_file = Path(__file__).resolve().parent / "model" / "weights" / "ensemble_pretrained.pt"
        if weights_file.exists():
            try:
                ckpt = torch.load(weights_file, map_location="cpu")
                if ckpt.get("state_dim") == config.model.state_dim:
                    self.bull_net.load_state_dict(ckpt["bull_state_dict"])
                    self.bear_net.load_state_dict(ckpt["bear_state_dict"])
                    self.arbiter_net.load_state_dict(ckpt["arbiter_state_dict"])
                    num_theories = len(ckpt.get("metadata", {}).get("theories", []))
                    print(f"[Bot Init] Loaded pre-trained ensemble checkpoint ({num_theories} quant theories active) from: {weights_file.name}")
                else:
                    print(f"[Bot Init] Checkpoint state_dim mismatch, running warm-start pretraining...")
                    run_pretraining(self.bull_net, self.bear_net, self.arbiter_net, epochs=10, state_dim=config.model.state_dim)
            except Exception as e:
                print(f"[Bot Init] Failed loading checkpoint ({e}), running warm-start pretraining...")
                run_pretraining(self.bull_net, self.bear_net, self.arbiter_net, epochs=10, state_dim=config.model.state_dim)
        else:
            print("[Bot Init] Pre-training neural debate ensemble on warm-start buffer...")
            run_pretraining(self.bull_net, self.bear_net, self.arbiter_net, epochs=15, state_dim=config.model.state_dim)
        print("[Bot Init] Ensemble pre-trained and ready.")

    def run_cycle(self) -> CycleSummary:
        """Executes one complete autonomous discovery, debate, risk check, and execution cycle."""
        start_time = time.time()
        print("\n" + "=" * 65)
        print(f"[Cycle Start] Mode: {'DRY-RUN' if self.dry_run else 'PAPER TRADING'}")

        # 1. Telemetry & Account Status
        try:
            account: AccountStatus = self.broker.get_account()
            positions: list[PositionInfo] = self.broker.get_positions()
        except Exception as e:
            print(f"[Error] Failed to fetch account telemetry: {e}")
            account = AccountStatus("ERR", "ACTIVE", 100_000.0, 100_000.0, 200_000.0, True)
            positions = []

        held_tickers = {p.ticker for p in positions}
        breaker_state = self.circuit_breaker.update_equity(account.equity)
        print(f"[Account] Equity: ${account.equity:,.2f} | Cash: ${account.cash:,.2f} | Active Positions: {len(positions)}")
        print(f"[Circuit Breaker] Status: {breaker_state.status.value} | {breaker_state.message}")

        if breaker_state.status == BreakerStatus.EMERGENCY:
            print("[CRITICAL] Emergency liquidation threshold breached! Halting and closing open positions.")
            for p in positions:
                self.broker.submit_order(OrderDirective(
                    ticker=p.ticker,
                    side=OrderSide.SELL,
                    shares=0,
                    limit_price=p.current_price,
                    stop_loss_price=0.0,
                    dollar_allocation=0.0,
                    risk_pct_actual=0.0,
                    is_authorized=True
                ))
            return CycleSummary(start_time, 0, 0, 0, 0, breaker_state.status.value, account.equity)

        # 1.5 Active Position Management (Monitor currently open positions)
        if positions:
            print(f"[Position Monitor] Actively managing {len(positions)} open positions...")
            for pos in positions:
                stock_data = self.screener.screen_ticker(pos.ticker)
                if stock_data:
                    port_ctx = PortfolioContext(
                        cash_ratio=(account.cash / account.equity) if account.equity > 0 else 1.0,
                        current_stock_weight=(pos.market_value / account.equity) if account.equity > 0 else 0.0,
                        unrealized_pnl_pct=pos.unrealized_pnl_pct,
                        daily_drawdown_pct=breaker_state.daily_drawdown_pct
                    )
                    s_tensor = self.normalizer.build_state_vector(
                        stock_data.technicals,
                        SentimentMetrics(),
                        stock_data.fundamentals,
                        stock_data.microstructure,
                        port_ctx,
                        DebateMetrics()
                    )
                    v_pos = self.arena.evaluate_debate(s_tensor)

                    entry_px = pos.current_price / (1.0 + pos.unrealized_pnl_pct) if (1.0 + pos.unrealized_pnl_pct) > 0 else pos.current_price
                    base_stop = max(0.01, entry_px - 2.0 * stock_data.technicals.atr)

                    pos_directive = self.governor.evaluate_held_position(
                        ticker=pos.ticker,
                        shares=pos.shares,
                        entry_price=entry_px,
                        current_price=pos.current_price,
                        atr=stock_data.technicals.atr,
                        unrealized_pnl_pct=pos.unrealized_pnl_pct,
                        debate_action=v_pos.chosen_action,
                        current_stop_price=base_stop
                    )

                    if pos_directive.action_type == PositionActionType.EXIT:
                        print(f"[Position Monitor: EXIT TRIGGERED] {pos.ticker} - {pos_directive.reason}")
                        exit_res = self.broker.submit_order(OrderDirective(
                            ticker=pos.ticker,
                            side=OrderSide.SELL,
                            shares=0,
                            limit_price=pos.current_price,
                            stop_loss_price=0.0,
                            dollar_allocation=0.0,
                            risk_pct_actual=0.0,
                            is_authorized=True
                        ))
                        if exit_res.success:
                            print(f"[Broker: EXIT SUCCESS] Closed position in {pos.ticker}")
                    elif pos_directive.action_type == PositionActionType.UPDATE_STOP:
                        print(f"[Position Monitor: RATCHET STOP] {pos.ticker} - {pos_directive.reason}")
                        updated = self.broker.update_stop_loss(pos.ticker, pos_directive.new_stop_price)
                        if updated:
                            print(f"[Broker: STOP UPDATED] {pos.ticker} stop ratcheted to ${pos_directive.new_stop_price:.2f}")
                    else:
                        print(f"[Position Monitor: HOLD] {pos.ticker} ({pos.shares} shs @ ${pos.current_price:.2f}, PnL: {pos.unrealized_pnl_pct*100:+.2f}%)")

        # 2. Open-Ended Internet Discovery (agent-reach / RSS)
        print("[Discovery] Scanning financial feeds and market news for high-conviction catalysts...")
        discovered = self.scout.scan_market_feeds()
        print(f"[Discovery] Found {len(discovered)} ticker candidates: {[c.ticker for c in discovered[:6]]}")

        for d in discovered[:10]:
            self.ledger.log_discovery(d.ticker, d.mention_count, d.sentiment_score, d.catalysts)

        # 3. Quantitative Screening Gate
        candidate_tickers = [d.ticker for d in discovered[:8]]
        print(f"[Screener] Filtering {len(candidate_tickers)} candidates for liquidity (ADV>$10M, Cap>$500M, P>$5)...")
        screened_stocks = self.screener.screen_batch(candidate_tickers)
        print(f"[Screener] Approved {len(screened_stocks)} liquid stocks: {[s.ticker for s in screened_stocks]}")

        authorized_count = 0
        executed_count = 0

        # 4. Neural Debate & Risk Execution for each approved stock
        for stock in screened_stocks:
            mention = next((m for m in discovered if m.ticker == stock.ticker), None)
            sent_score = mention.sentiment_score if mention else 0.0
            mention_cnt = mention.mention_count if mention else 1

            sent_metrics = SentimentMetrics(
                mention_count=mention_cnt,
                sentiment_score=sent_score,
                catalyst_urgency=0.7,
                source_credibility=0.85
            )

            port_context = PortfolioContext(
                cash_ratio=(account.cash / account.equity) if account.equity > 0 else 1.0,
                current_stock_weight=0.0,
                unrealized_pnl_pct=0.0,
                daily_drawdown_pct=breaker_state.daily_drawdown_pct
            )

            # Build initial state with neutral debate prior
            state_tensor = self.normalizer.build_state_vector(
                stock.technicals,
                sent_metrics,
                stock.fundamentals,
                stock.microstructure,
                port_context,
                DebateMetrics()
            )

            # Run Debate between Bull and Bear
            verdict: DebateVerdict = self.arena.evaluate_debate(state_tensor)
            self.ledger.log_debate(
                ticker=stock.ticker,
                bull_conf=verdict.bull_probs[1] + verdict.bull_probs[2],
                bear_conf=verdict.bear_probs[3],
                disagreement_var=verdict.disagreement_variance,
                action=verdict.chosen_action,
                is_approved=verdict.is_consensus_approved
            )

            action_names = {0: "HOLD", 1: "BUY (0.5x)", 2: "BUY (1.0x)", 3: "EXIT"}
            print(
                f"[Debate: {stock.ticker}] Action: {action_names.get(verdict.chosen_action)} | "
                f"Conf: {verdict.action_confidence*100:.1f}% | "
                f"Disagreement Var: {verdict.disagreement_variance:.4f} | "
                f"Consensus: {'APPROVED' if verdict.is_consensus_approved else 'REJECTED (High Uncertainty)'}"
            )

            # 5. Risk Governor Evaluation
            directive: OrderDirective = self.governor.evaluate_order(
                ticker=stock.ticker,
                action=verdict.chosen_action,
                current_price=stock.price,
                atr=stock.technicals.atr,
                portfolio_equity=account.equity,
                available_cash=account.cash,
                current_positions_count=len(positions),
                is_already_held=(stock.ticker in held_tickers),
                circuit_breaker_active=(breaker_state.status != BreakerStatus.NORMAL)
            )

            if directive.is_authorized:
                authorized_count += 1
                print(
                    f"[Risk Governor: APPROVED] {directive.side.value.upper()} {directive.shares} shares of {directive.ticker} "
                    f"at ~${directive.limit_price:.2f} (Stop Loss: ${directive.stop_loss_price:.2f}, Capital: ${directive.dollar_allocation:,.2f})"
                )

                # 6. Broker Execution
                res = self.broker.submit_order(directive)
                if res.success:
                    executed_count += 1
                    print(f"[Broker: SUCCESS] {res.message} (Order ID: {res.order_id})")
                    self.ledger.log_trade(TradeRecord(
                        ticker=directive.ticker,
                        side=directive.side.value,
                        shares=directive.shares,
                        price=directive.limit_price,
                        stop_loss=directive.stop_loss_price,
                        order_id=res.order_id or "UNKNOWN",
                        status="SUBMITTED",
                        realized_pnl=0.0
                    ))
                else:
                    print(f"[Broker: REJECTED] {res.message}")
            else:
                if verdict.chosen_action != 0:
                    print(f"[Risk Governor: BLOCKED] {stock.ticker} - {directive.rejection_reason}")

            # Store transition in replay buffer for continuous self-learning
            reward_estimate = float(stock.technicals.macd_hist * 0.01)
            self.buffer.push(
                state=state_tensor,
                action=verdict.chosen_action,
                reward=reward_estimate,
                next_state=state_tensor,
                done=False,
                disagreement_var=verdict.disagreement_variance
            )

        # 7. Post-cycle online learning step
        if len(self.buffer) >= self.config.model.batch_size:
            metrics = self.trainer.train_step(self.buffer, batch_size=self.config.model.batch_size)
            print(f"[Self-Learning] Online training step {metrics.total_steps} completed | Bull Loss: {metrics.bull_loss:.4f} | Bear Loss: {metrics.bear_loss:.4f}")

        elapsed = time.time() - start_time
        print(f"[Cycle Finished] Completed in {elapsed:.2f}s | Orders Authorized: {authorized_count} | Executed: {executed_count}")
        print("=" * 65 + "\n")

        return CycleSummary(
            timestamp=start_time,
            discovered_count=len(discovered),
            screened_count=len(screened_stocks),
            orders_authorized=authorized_count,
            orders_executed=executed_count,
            circuit_breaker_status=breaker_state.status.value,
            total_equity=account.equity
        )
