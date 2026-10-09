"""CLI Entrypoint for the Neuro-Symbolic Trading Bot.

Usage:
    python main.py --mode paper --cycles 1
    python main.py --dry-run
    python main.py --status
    python main.py --discover
"""

from __future__ import annotations
import argparse
import sys
import time
from config import load_config
from bot import TradingBot
from broker.alpaca_client import AlpacaPaperClient
from broker.ledger import TradeLedger
from discovery.scout import DiscoveryScout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Agent-Reach + Multi-Neural-Network Debate Trading Bot")
    parser.add_argument("--mode", choices=["paper", "dry-run"], default="paper", help="Execution mode (default: paper)")
    parser.add_argument("--dry-run", action="store_true", help="Shortcut for --mode dry-run (simulation without real orders)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of autonomous scanning cycles to run (default: 1)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds to sleep between cycles if cycles > 1 (default: 60)")
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
        print(f"Status:        {acc.status}")
        print(f"Paper Trading: {acc.is_paper}")
        print(f"Total Equity:  ${acc.equity:,.2f}")
        print(f"Cash Balance:  ${acc.cash:,.2f}")
        print(f"Buying Power:  ${acc.buying_power:,.2f}")
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

    for i in range(args.cycles):
        print(f"\n>>> Running Autonomous Cycle {i + 1} of {args.cycles} <<<")
        bot.run_cycle()
        if i < args.cycles - 1:
            print(f"Sleeping {args.interval}s until next cycle...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
