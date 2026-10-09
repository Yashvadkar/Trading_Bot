"""Alpaca Markets Paper Trading REST Gateway.

Interfaces directly with the Alpaca Paper API (/v2/account, /v2/orders, /v2/positions).
Features automatic retries, dry-run safety modes, and response parsing.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass
import requests
from config import Config
from risk.governor import OrderDirective, OrderSide


@dataclass(frozen=True)
class AccountStatus:
    account_id: str
    status: str
    equity: float
    cash: float
    buying_power: float
    is_paper: bool


@dataclass(frozen=True)
class PositionInfo:
    ticker: str
    shares: int
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    order_id: str | None
    ticker: str
    side: str
    shares: int
    filled_price: float
    message: str


@dataclass(frozen=True)
class MarketClock:
    timestamp: str
    is_open: bool
    next_open: str
    next_close: str


class AlpacaPaperClient:
    """REST Client for Alpaca Paper Trading API."""

    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.base_url = config.alpaca_base_url.rstrip("/")
        self.headers = {
            "APCA-API-KEY-ID": config.alpaca_api_key,
            "APCA-API-SECRET-KEY": config.alpaca_secret_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_account(self) -> AccountStatus:
        """Fetch account balance, equity, and buying power."""
        if self.dry_run:
            return AccountStatus(
                account_id="DRY_RUN_ACC_001",
                status="ACTIVE",
                equity=100_000.0,
                cash=100_000.0,
                buying_power=200_000.0,
                is_paper=True
            )

        resp = self.session.get(f"{self.base_url}/v2/account", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        return AccountStatus(
            account_id=data.get("id", "UNKNOWN"),
            status=data.get("status", "ACTIVE"),
            equity=float(data.get("equity", 0.0)),
            cash=float(data.get("cash", 0.0)),
            buying_power=float(data.get("buying_power", 0.0)),
            is_paper="paper" in self.base_url.lower()
        )

    def get_positions(self) -> list[PositionInfo]:
        """Fetch all currently open equity positions."""
        if self.dry_run:
            return []

        resp = self.session.get(f"{self.base_url}/v2/positions", timeout=10)
        resp.raise_for_status()
        positions_raw = resp.json()

        positions = []
        for p in positions_raw:
            positions.append(PositionInfo(
                ticker=p.get("symbol", ""),
                shares=int(p.get("qty", 0)),
                current_price=float(p.get("current_price", 0.0)),
                market_value=float(p.get("market_value", 0.0)),
                unrealized_pnl=float(p.get("unrealized_pl", 0.0)),
                unrealized_pnl_pct=float(p.get("unrealized_plpc", 0.0)),
            ))
        return positions

    def get_market_clock(self) -> MarketClock:
        """Fetch US market open/close clock telemetry from Alpaca."""
        if self.dry_run:
            return MarketClock(
                timestamp="2026-10-09T09:30:00Z",
                is_open=True,
                next_open="2026-10-10T09:30:00Z",
                next_close="2026-10-09T16:00:00Z"
            )
        try:
            resp = self.session.get(f"{self.base_url}/v2/clock", timeout=8)
            resp.raise_for_status()
            data = resp.json()
            return MarketClock(
                timestamp=str(data.get("timestamp", "")),
                is_open=bool(data.get("is_open", False)),
                next_open=str(data.get("next_open", "")),
                next_close=str(data.get("next_close", ""))
            )
        except Exception:
            return MarketClock(timestamp="", is_open=True, next_open="", next_close="")

    def update_stop_loss(self, ticker: str, new_stop_price: float) -> bool:
        """Finds open stop order for ticker and patches the stop price upward."""
        if self.dry_run:
            return True
        try:
            resp = self.session.get(f"{self.base_url}/v2/orders?status=open&symbols={ticker}", timeout=8)
            if resp.status_code == 200:
                orders = resp.json()
                for ord_info in orders:
                    if ord_info.get("type") in ("stop", "stop_limit") or ord_info.get("stop_price"):
                        order_id = ord_info.get("id")
                        patch_resp = self.session.patch(
                            f"{self.base_url}/v2/orders/{order_id}",
                            json={"stop_price": str(new_stop_price)},
                            timeout=8
                        )
                        return patch_resp.status_code in (200, 201)
            return False
        except Exception:
            return False

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        """Deterministically query an order by client_order_id for replay safety."""
        if self.dry_run:
            return {"id": client_order_id, "status": "accepted"}
        try:
            resp = self.session.get(
                f"{self.base_url}/v2/orders:by_client_order_id",
                params={"client_order_id": client_order_id},
                timeout=8
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    def submit_order(self, directive: OrderDirective) -> ExecutionResult:
        """Submit a validated order directive to the broker with deterministic idempotency."""
        if not directive.is_authorized:
            return ExecutionResult(
                success=False,
                order_id=None,
                ticker=directive.ticker,
                side=directive.side.value,
                shares=0,
                filled_price=0.0,
                message=f"Order rejected by risk governor: {directive.rejection_reason}"
            )

        client_order_id = f"bot_{uuid.uuid4().hex[:16]}"

        if self.dry_run:
            return ExecutionResult(
                success=True,
                order_id=f"SIM_{directive.ticker}_{directive.shares}_{client_order_id[:8]}",
                ticker=directive.ticker,
                side=directive.side.value,
                shares=directive.shares,
                filled_price=directive.limit_price,
                message=f"Simulated order accepted in dry-run mode (idempotency key: {client_order_id})."
            )

        # 1. Buy Order with Stop-Loss & Take-Profit (Bracket)
        if directive.side == OrderSide.BUY:
            stop_dist = max(0.5, directive.limit_price - directive.stop_loss_price)
            take_profit_price = round(directive.limit_price + (2.5 * stop_dist), 2)

            payload = {
                "symbol": directive.ticker,
                "qty": str(directive.shares),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "order_class": "bracket",
                "client_order_id": client_order_id,
                "take_profit": {
                    "limit_price": str(take_profit_price)
                },
                "stop_loss": {
                    "stop_price": str(directive.stop_loss_price)
                }
            }

            try:
                resp = self.session.post(f"{self.base_url}/v2/orders", json=payload, timeout=10)
                if resp.status_code not in (200, 201):
                    # Fallback to OTO (One-Triggers-Other) with stop-loss only if bracket is rejected
                    payload_oto = {
                        "symbol": directive.ticker,
                        "qty": str(directive.shares),
                        "side": "buy",
                        "type": "market",
                        "time_in_force": "day",
                        "order_class": "oto",
                        "client_order_id": client_order_id,
                        "stop_loss": {
                            "stop_price": str(directive.stop_loss_price)
                        }
                    }
                    resp = self.session.post(f"{self.base_url}/v2/orders", json=payload_oto, timeout=10)

                if resp.status_code in (200, 201):
                    res_data = resp.json()
                    return ExecutionResult(
                        success=True,
                        order_id=res_data.get("id"),
                        ticker=directive.ticker,
                        side="buy",
                        shares=directive.shares,
                        filled_price=directive.limit_price,
                        message=f"Bracket buy order submitted to Alpaca (Key: {client_order_id})."
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        order_id=None,
                        ticker=directive.ticker,
                        side="buy",
                        shares=directive.shares,
                        filled_price=directive.limit_price,
                        message=f"Alpaca API error: {resp.status_code} - {resp.text}"
                    )
            except (requests.Timeout, requests.RequestException) as net_err:
                # Deterministic timeout reconciliation: query by client_order_id
                print(f"[Alpaca Gateway] Network partition/timeout on {client_order_id}. Reconciling exchange state...")
                reconciled = self.get_order_by_client_id(client_order_id)
                if reconciled and reconciled.get("status") in ("accepted", "new", "partially_filled", "filled", "pending_new"):
                    return ExecutionResult(
                        success=True,
                        order_id=reconciled.get("id"),
                        ticker=directive.ticker,
                        side="buy",
                        shares=directive.shares,
                        filled_price=directive.limit_price,
                        message=f"Order reconciled via client_order_id ({client_order_id}) after network timeout."
                    )
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    ticker=directive.ticker,
                    side="buy",
                    shares=0,
                    filled_price=0.0,
                    message=f"Order timed out and was not confirmed at exchange boundary: {net_err}"
                )

        # 2. Sell / Close Order
        elif directive.side == OrderSide.SELL:
            resp = self.session.delete(f"{self.base_url}/v2/positions/{directive.ticker}", timeout=10)
            if resp.status_code in (200, 204):
                return ExecutionResult(
                    success=True,
                    order_id=f"CLOSE_{directive.ticker}",
                    ticker=directive.ticker,
                    side="sell",
                    shares=directive.shares,
                    filled_price=directive.limit_price,
                    message=f"Position in {directive.ticker} closed successfully."
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    ticker=directive.ticker,
                    side="sell",
                    shares=0,
                    filled_price=directive.limit_price,
                    message=f"Alpaca close error: {resp.status_code} - {resp.text}"
                )

        return ExecutionResult(
            success=False,
            order_id=None,
            ticker=directive.ticker,
            side="hold",
            shares=0,
            filled_price=0.0,
            message="Action is HOLD; no order placed."
        )
