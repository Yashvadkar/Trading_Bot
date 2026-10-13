"""CLI Entrypoint for the Neuro-Symbolic Trading Bot.

Supports:
    python main.py --mode paper --cycles 5
    python main.py --daemon
    python main.py --dry-run
    python main.py --status
    python main.py --discover
"""

from __future__ import annotations
import argparse
import signal
import sys
import time
from config import load_config, Config
from bot import TradingBot
from broker.alpaca_client import AlpacaPaperClient, MarketClock
from broker.ledger import TradeLedger
from discovery.scout import DiscoveryScout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Agent-Reach + Multi-Neural-Network Debate Trading Bot")
    parser.add_argument("--mode", choices=["paper", "dry-run"], default="paper", help="Execution mode (default: paper)")
    parser.add_argument("--dry-run", action="store_true", help="Shortcut for --mode dry-run (simulation without real orders)")
    parser.add_argument("--daemon", action="store_true", help="Run 24/7 continuous autonomous loop with market clock awareness")
    parser.add_argument("--continuous", action="store_true", help="Alias for --daemon")
    parser.add_argument("--cycles", type=int, default=1, help="Number of autonomous scanning cycles to run if not in daemon mode (default: 1)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds to sleep between cycles (default: 60)")
    parser.add_argument("--status", action="store_true", help="Display account equity, positions, and recent trades")
    parser.add_argument("--discover", action="store_true", help="Scan internet feeds for stocks without trading")
    return parser.parse_args()


def display_status() -> None:
    cfg = load_config()
    client = AlpacaPaperClient(cfg, dry_run=False)
    ledger = TradeLedger()

    print("\n" + "=" * 55)
    print("           ALPACA ACCOUNT TELEMETRY")
    print("=" * 55)
    try:
        acc = client.get_account()
        clock: MarketClock = client.get_market_clock()
        print(f"Status:        {acc.status}")
        print(f"Paper Trading: {acc.is_paper}")
        print(f"Total Equity:  ${acc.equity:,.2f}")
        print(f"Cash Balance:  ${acc.cash:,.2f}")
        print(f"Buying Power:  ${acc.buying_power:,.2f}")
        print(f"Market Open:   {clock.is_open} (Next Open: {clock.next_open})")
    except Exception as e:
        print(f"Failed to query Alpaca API: {e}")

    print("\n" + "=" * 55)
    print("           CURRENT OPEN POSITIONS")
    print("=" * 55)
    try:
        positions = client.get_positions()
        if not positions:
            print("No open positions.")
        else:
            for p in positions:
                print(f"- {p.ticker}: {p.shares} shares @ ${p.current_price:.2f} | Unrealized PnL: ${p.unrealized_pnl:.2f} ({p.unrealized_pnl_pct*100:.2f}%)")
    except Exception as e:
        print(f"Failed to query positions: {e}")

    print("\n" + "=" * 55)
    print("           RECENT AUDIT LEDGER TRADES")
    print("=" * 55)
    trades = ledger.get_recent_trades(limit=5)
    if not trades:
        print("No recent trades logged in SQLite database.")
    else:
        for t in trades:
            print(f"[{t['timestamp'][:19]}] {t['ticker']} {t['side'].upper()} {t['shares']} shares @ ${t['price']:.2f} | Status: {t['status']}")
    print("=" * 55 + "\n")


def display_discovery() -> None:
    print("\n[Discovery] Scanning financial feeds and market news via agent-reach...")
    scout = DiscoveryScout()
    discovered = scout.scan_market_feeds()
    print(f"[Discovery] Found {len(discovered)} ticker mentions:\n")
    for d in discovered:
        print(f"  • ${d.ticker:<6} | Mentions: {d.mention_count:<2} | Sentiment: {d.sentiment_score:+0.2f} | Catalysts: {', '.join(d.catalysts) if d.catalysts else 'None'}")
    print()


def run_daemon_loop(bot: TradingBot, config: Config, interval: int) -> None:
    """Continuous 24/7 loop with market hours awareness and graceful signal handling."""
    stop_requested = False

    def handle_signal(sig, frame):
        nonlocal stop_requested
        print("\n[Daemon] Graceful shutdown requested (Ctrl+C). Finishing current loop...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n" + "#" * 65)
    print("  STARTING 24/7 AUTONOMOUS DAEMON TRADING ENGINE")
    print("  Active Scanning Interval: " + str(interval) + "s")
    print("  Press Ctrl+C to stop safely at any time.")
    print("#" * 65 + "\n")

    cycle_count = 0
    trained_off_hours_today = False

    while not stop_requested:
        cycle_count += 1
        clock: MarketClock = bot.broker.get_market_clock()

        if clock.is_open:
            trained_off_hours_today = False  # Reset for next close
            print(f"\n[Daemon] Market is OPEN. Executing Cycle #{cycle_count}...")
            cycle_start = time.time()
            bot.run_cycle()
            cycle_duration = time.time() - cycle_start

            # Candle-aligned or drift-compensated sleep cadence
            if interval >= 60:
                # Align to top-of-minute candle boundaries (:00s)
                next_boundary = (int(time.time()) // interval + 1) * interval
                sleep_needed = max(1.0, float(next_boundary - time.time()))
            else:
                sleep_needed = max(0.5, float(interval - cycle_duration))

            print(f"[Daemon] Cycle duration: {cycle_duration:.2f}s | Target sleep: {sleep_needed:.1f}s")
            sleep_remaining = sleep_needed
            while sleep_remaining > 0 and not stop_requested:
                step = min(1.0, sleep_remaining)
                time.sleep(step)
                sleep_remaining -= step
        else:
            print(f"\n[Daemon] Market is CLOSED. (Next Open: {clock.next_open})")
            if not trained_off_hours_today and len(bot.buffer) >= bot.config.model.batch_size:
                print("[Daemon: Off-Hours] Executing post-market neural co-training update...")
                metrics = bot.trainer.train_step(bot.buffer, batch_size=bot.config.model.batch_size)
                print(f"[Daemon: Off-Hours] Training step completed | Bull Loss: {metrics.bull_loss:.4f} | Bear Loss: {metrics.bear_loss:.4f}")
                trained_off_hours_today = True

            # Standby sleep during off hours
            standby_seconds = min(config.daemon.off_hours_sleep, 300)
            print(f"[Daemon] Entering standby sleep ({standby_seconds}s)...")
            sleep_remaining = standby_seconds
            while sleep_remaining > 0 and not stop_requested:
                step = min(2, sleep_remaining)
                time.sleep(step)
                sleep_remaining -= step

    print("[Daemon] Clean shutdown complete. Final telemetry saved to SQLite ledger.")


def main() -> None:
    args = parse_args()

    if args.status:
        display_status()
        sys.exit(0)

    if args.discover:
        display_discovery()
        sys.exit(0)

    is_dry_run = args.dry_run or (args.mode == "dry-run")
    config = load_config()
    print(config.summary())

    bot = TradingBot(config=config, dry_run=is_dry_run)
    try:
        if args.daemon or args.continuous:
            run_daemon_loop(bot, config, args.interval)
        else:
            for i in range(args.cycles):
                print(f"\n>>> Running Autonomous Cycle {i + 1} of {args.cycles} <<<")
                bot.run_cycle()
                if i < args.cycles - 1:
                    print(f"Sleeping {args.interval}s until next cycle...")
                    time.sleep(args.interval)
    finally:
        bot.close()


if __name__ == "__main__":
    main()
