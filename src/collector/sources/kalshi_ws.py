"""Kalshi WebSocket client — real-time orderbook, ticker, and trade events."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Callable

import websockets

from collector.config import Config
from collector.sources.kalshi_auth import load_private_key, make_ws_auth_headers
from collector.types import MarketSnapshot

log = logging.getLogger(__name__)

BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 30.0
BACKOFF_MULTIPLIER = 2.0


def _dollars_to_cents(s: str) -> int:
    """Convert Kalshi dollar string to integer cents. '0.2200' -> 22"""
    return round(float(s) * 100)


def _dollars_to_contracts(s: str) -> int:
    """Convert Kalshi dollar-denominated quantity to contracts. '500.00' -> 500"""
    return round(float(s))


class OrderbookState:
    """Maintains local orderbook state from Kalshi v2 WS snapshots + deltas.

    Kalshi v2 WS format:
      Snapshot: yes_dollars_fp / no_dollars_fp = [["price_dollars", "qty_dollars"], ...]
      Delta: {side: "yes"|"no", price_dollars: "0.22", delta_fp: "100.00"}

    We store internally as {price_cents: qty_contracts}.
    """

    def __init__(self):
        self.yes_bids: dict[str, dict[int, int]] = {}  # ticker -> {price_cents: qty}
        self.no_bids: dict[str, dict[int, int]] = {}
        self._has_snapshot: set[str] = set()

    def apply_snapshot(self, ticker: str, data: dict) -> None:
        """Apply a full orderbook snapshot from Kalshi v2 WS."""
        yes_book: dict[int, int] = {}
        for entry in (data.get("yes_dollars_fp") or []):
            price_cents = _dollars_to_cents(entry[0])
            qty = _dollars_to_contracts(entry[1])
            yes_book[price_cents] = qty

        no_book: dict[int, int] = {}
        for entry in (data.get("no_dollars_fp") or []):
            price_cents = _dollars_to_cents(entry[0])
            qty = _dollars_to_contracts(entry[1])
            no_book[price_cents] = qty

        self.yes_bids[ticker] = yes_book
        self.no_bids[ticker] = no_book
        self._has_snapshot.add(ticker)

    def apply_delta(self, ticker: str, data: dict) -> None:
        """Apply an incremental orderbook delta from Kalshi v2 WS."""
        if ticker not in self._has_snapshot:
            return

        side = data.get("side", "")
        price_str = data.get("price_dollars", "")
        delta_str = data.get("delta_fp", "")
        if not price_str or not delta_str:
            return

        price_cents = _dollars_to_cents(price_str)
        delta = _dollars_to_contracts(delta_str)

        book = self.yes_bids if side == "yes" else self.no_bids
        book_for_ticker = book.get(ticker, {})

        current = book_for_ticker.get(price_cents, 0)
        new_qty = current + delta
        if new_qty <= 0:
            book_for_ticker.pop(price_cents, None)
        else:
            book_for_ticker[price_cents] = new_qty

        book[ticker] = book_for_ticker

    def has_snapshot(self, ticker: str) -> bool:
        return ticker in self._has_snapshot

    def get_best_prices(self, ticker: str) -> dict:
        """Get best bid/ask + liquidity for a ticker in cents."""
        yes_b = self.yes_bids.get(ticker, {})
        no_b = self.no_bids.get(ticker, {})

        best_yes_bid = max(yes_b.keys()) if yes_b else None
        best_no_bid = max(no_b.keys()) if no_b else None
        best_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
        best_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None

        # Liquidity at best price
        yes_bid_qty = yes_b.get(best_yes_bid, 0) if best_yes_bid is not None else 0
        no_bid_qty = no_b.get(best_no_bid, 0) if best_no_bid is not None else 0
        yes_ask_qty = no_bid_qty   # yes ask filled by no bids
        no_ask_qty = yes_bid_qty   # no ask filled by yes bids

        total_yes_depth = sum(yes_b.values())
        total_no_depth = sum(no_b.values())
        spread = (best_yes_ask - best_yes_bid) if (best_yes_ask is not None and best_yes_bid is not None) else None

        return {
            "best_yes_bid_cents": best_yes_bid,
            "best_yes_ask_cents": best_yes_ask,
            "best_no_bid_cents": best_no_bid,
            "best_no_ask_cents": best_no_ask,
            "yes_bid_qty": yes_bid_qty,
            "yes_ask_qty": yes_ask_qty,
            "no_bid_qty": no_bid_qty,
            "no_ask_qty": no_ask_qty,
            "total_yes_depth": total_yes_depth,
            "total_no_depth": total_no_depth,
            "spread_cents": spread,
            "num_yes_levels": len(yes_b),
            "num_no_levels": len(no_b),
        }

    def clear_ticker(self, ticker: str) -> None:
        self.yes_bids.pop(ticker, None)
        self.no_bids.pop(ticker, None)
        self._has_snapshot.discard(ticker)


class KalshiWS:
    """Kalshi WebSocket client for real-time market data."""

    def __init__(
        self,
        cfg: Config,
        on_snapshot: Callable[[str, dict], None] | None = None,
    ):
        self._cfg = cfg
        self._private_key = load_private_key(cfg.kalshi.private_key_path)
        self._on_snapshot = on_snapshot  # (ticker, best_prices)
        self._orderbook = OrderbookState()
        self._ticker_data: dict[str, dict] = {}  # ticker -> {last_price_cents, volume}
        self._subscribed_tickers: set[str] = set()
        self._desired_tickers: set[str] = set()
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._cmd_id = 0
        self._seq_by_sid: dict[int, int] = {}
        self._running = False

    @property
    def orderbook(self) -> OrderbookState:
        return self._orderbook

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket connection is active."""
        return self._ws is not None

    def set_snapshot_callback(
        self, callback: Callable[[str, dict], None]
    ) -> None:
        """Set the callback for orderbook snapshot/delta updates."""
        self._on_snapshot = callback

    def set_tickers(self, tickers: set[str]) -> None:
        """Update the set of tickers we want to track."""
        self._desired_tickers = tickers

    async def start(self) -> None:
        """Run the WebSocket listener with auto-reconnect."""
        self._running = True
        backoff = BACKOFF_INITIAL

        while self._running:
            try:
                await self._connect()
                backoff = BACKOFF_INITIAL
            except asyncio.CancelledError:
                self._running = False
                return
            except Exception:
                log.exception("KalshiWS error, reconnecting in %.1fs", backoff)
                self._mark_stale()
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX)

    def stop(self) -> None:
        self._running = False

    async def _connect(self) -> None:
        headers = make_ws_auth_headers(
            self._cfg.kalshi.api_key_id, self._private_key
        )
        url = self._cfg.kalshi.ws_url
        log.info("KalshiWS connecting")

        async with websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=self._cfg.kalshi.ws_ping_secs,
            ping_timeout=self._cfg.kalshi.ws_ping_timeout_secs,
        ) as ws:
            self._ws = ws
            self._subscribed_tickers.clear()
            self._seq_by_sid.clear()
            log.info("KalshiWS connected")

            # Subscribe to desired tickers
            if self._desired_tickers:
                await self._subscribe(self._desired_tickers)

            async for raw in ws:
                msg = json.loads(raw)
                await self._handle_message(msg)

    async def _subscribe(self, tickers: set[str]) -> None:
        if not tickers or not self._ws:
            return
        self._cmd_id += 1
        cmd = {
            "id": self._cmd_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "ticker", "trade"],
                "market_tickers": list(tickers),
            },
        }
        await self._ws.send(json.dumps(cmd))
        self._subscribed_tickers.update(tickers)

    async def _unsubscribe(self, tickers: set[str]) -> None:
        if not tickers or not self._ws:
            return
        self._cmd_id += 1
        cmd = {
            "id": self._cmd_id,
            "cmd": "unsubscribe",
            "params": {
                "channels": ["orderbook_delta", "ticker", "trade"],
                "market_tickers": list(tickers),
            },
        }
        await self._ws.send(json.dumps(cmd))
        self._subscribed_tickers -= tickers

    async def sync_subscriptions(self) -> None:
        """Sync subscriptions to match desired tickers."""
        to_add = self._desired_tickers - self._subscribed_tickers
        to_remove = self._subscribed_tickers - self._desired_tickers
        if to_remove:
            await self._unsubscribe(to_remove)
            for t in to_remove:
                self._orderbook.clear_ticker(t)
        if to_add:
            await self._subscribe(to_add)

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        sid = msg.get("sid")
        seq = msg.get("seq")

        # Track sequence numbers
        if sid is not None and seq is not None:
            prev = self._seq_by_sid.get(sid, 0)
            if seq > prev + 100:
                log.warning("KalshiWS sequence gap: sid=%d prev=%d new=%d", sid, prev, seq)
            self._seq_by_sid[sid] = seq

        if msg_type == "orderbook_snapshot":
            ticker = msg.get("msg", {}).get("market_ticker", "")
            if ticker:
                self._orderbook.apply_snapshot(ticker, msg.get("msg", {}))
                self._notify(ticker)

        elif msg_type == "orderbook_delta":
            ticker = msg.get("msg", {}).get("market_ticker", "")
            if ticker:
                self._orderbook.apply_delta(ticker, msg.get("msg", {}))
                self._notify(ticker)

        elif msg_type == "ticker":
            inner = msg.get("msg", {})
            ticker = inner.get("market_ticker", "")
            if ticker:
                td = self._ticker_data.setdefault(ticker, {})
                # Kalshi ticker fields: price_dollars, volume_fp, open_interest_fp
                price_str = inner.get("price_dollars")
                if price_str is not None:
                    try:
                        td["last_price_cents"] = _dollars_to_cents(str(price_str))
                    except (ValueError, TypeError):
                        pass
                vol_str = inner.get("volume_fp")
                if vol_str is not None:
                    try:
                        td["volume"] = _dollars_to_contracts(str(vol_str))
                    except (ValueError, TypeError):
                        pass
                oi_str = inner.get("open_interest_fp")
                if oi_str is not None:
                    try:
                        td["open_interest"] = _dollars_to_contracts(str(oi_str))
                    except (ValueError, TypeError):
                        pass
                # Trigger snapshot update with new ticker data
                self._notify(ticker)

        elif msg_type == "trade":
            inner = msg.get("msg", {})
            ticker = inner.get("market_ticker", "")
            if ticker:
                td = self._ticker_data.setdefault(ticker, {})
                # Kalshi trade fields: yes_price_dollars, no_price_dollars, count_fp
                trade_price = inner.get("yes_price_dollars", inner.get("no_price_dollars"))
                if trade_price is not None:
                    try:
                        td["last_price_cents"] = _dollars_to_cents(str(trade_price))
                    except (ValueError, TypeError):
                        pass
                count_str = inner.get("count_fp")
                if count_str is not None:
                    try:
                        td["volume"] = td.get("volume", 0) + _dollars_to_contracts(str(count_str))
                    except (ValueError, TypeError):
                        pass

        elif msg_type == "error":
            log.error("KalshiWS error: %s", msg)

    def _notify(self, ticker: str) -> None:
        if self._on_snapshot:
            prices = self._orderbook.get_best_prices(ticker)
            # Merge in ticker data (last_price, volume, open_interest)
            td = self._ticker_data.get(ticker, {})
            if td:
                prices.update(td)
            self._on_snapshot(ticker, prices)

    def _mark_stale(self) -> None:
        """Mark all orderbooks as stale on disconnect."""
        for t in list(self._subscribed_tickers):
            self._orderbook.clear_ticker(t)
        self._subscribed_tickers.clear()
