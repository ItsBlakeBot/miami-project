"""Kalshi REST client — market discovery + settlement checks."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import aiohttp

from collector.config import Config
from collector.sources.kalshi_auth import load_private_key, make_rest_headers
from collector.types import MarketSnapshot

log = logging.getLogger(__name__)


STRIKE_RANGE_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees)?)?\s*(?:to|-|through)\s*(\d+(?:\.\d+)?)"
)
STRIKE_ABOVE_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees)?)?\s*(?:or\s+above|or\s+higher|and\s+above|and\s+higher|above|higher)"
)
STRIKE_BELOW_RE = re.compile(
    r"(?i)(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees)?)?\s*(?:or\s+below|or\s+lower|and\s+below|and\s+lower|below|lower)"
)


def _to_cents(value: str | int | float | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value) * 100) if isinstance(value, str) else round(float(value))
    except (TypeError, ValueError):
        return None


def _to_contracts(value: str | int | float | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return None


def _event_ticker_from_ticker(ticker: str) -> str:
    parts = ticker.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return ticker


def _series_ticker_from_event_ticker(event_ticker: str) -> str:
    return event_ticker.split("-")[0] if event_ticker else ""


def _forecast_date_from_event_ticker(event_ticker: str) -> str | None:
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[1].upper(), "%y%b%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _extract_strikes_from_text(*texts: str | None) -> tuple[float | None, float | None]:
    combined = " | ".join(t for t in texts if t)
    if not combined:
        return None, None

    if match := STRIKE_RANGE_RE.search(combined):
        return float(match.group(1)), float(match.group(2))
    if match := STRIKE_ABOVE_RE.search(combined):
        return float(match.group(1)), None
    if match := STRIKE_BELOW_RE.search(combined):
        return None, float(match.group(1))
    return None, None


def _parse_ticker(ticker: str) -> dict:
    """Parse a Kalshi temperature ticker into analysis-friendly metadata."""
    event_ticker = _event_ticker_from_ticker(ticker)
    series_ticker = _series_ticker_from_event_ticker(event_ticker)
    result = {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "market_type": "",
        "forecast_date": _forecast_date_from_event_ticker(event_ticker),
        "floor_strike": None,
        "cap_strike": None,
    }

    upper = ticker.upper()
    if "HIGH" in series_ticker or "HIGH" in upper:
        result["market_type"] = "high"
    elif "LOW" in series_ticker or "LOW" in upper:
        result["market_type"] = "low"

    match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", upper)
    if match:
        side, value = match.group(1), float(match.group(2))
        if side == "T":
            result["floor_strike"] = value
        else:
            result["cap_strike"] = value

    return result


def _market_metadata_from_response(market: dict) -> dict:
    ticker = market.get("ticker", "")
    parsed = _parse_ticker(ticker)

    event_ticker = market.get("event_ticker") or parsed["event_ticker"]
    series_ticker = _series_ticker_from_event_ticker(event_ticker) or parsed["series_ticker"]
    forecast_date = (
        _forecast_date_from_event_ticker(event_ticker)
        or parsed.get("forecast_date")
        or (market.get("expiration_time") or "")[:10]
        or None
    )

    floor_strike, cap_strike = _extract_strikes_from_text(
        market.get("yes_sub_title"),
        market.get("no_sub_title"),
        market.get("subtitle"),
        market.get("title"),
    )
    if parsed.get("floor_strike") is not None:
        floor_strike = parsed["floor_strike"]
    if parsed.get("cap_strike") is not None:
        cap_strike = parsed["cap_strike"]

    market_type = parsed.get("market_type", "")
    if not market_type:
        title_blob = " ".join(
            part for part in [event_ticker, market.get("title"), market.get("subtitle")] if part
        ).upper()
        if "HIGH" in title_blob:
            market_type = "high"
        elif "LOW" in title_blob:
            market_type = "low"

    return {
        "ticker": ticker,
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "market_type": market_type,
        "forecast_date": forecast_date,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "last_price_cents": _to_cents(market.get("last_price_dollars")),
        "volume": _to_contracts(market.get("volume_fp")),
    }


class KalshiREST:
    """Kalshi REST API client for market discovery and settlement checks."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._private_key = load_private_key(cfg.kalshi.private_key_path)
        self._session: aiohttp.ClientSession | None = None
        self._market_metadata_by_ticker: dict[str, dict] = {}

    async def open(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session, "KalshiREST not opened"
        return self._session

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        return make_rest_headers(
            self._cfg.kalshi.api_key_id, self._private_key, method, path
        )

    def get_market_metadata(self, ticker: str) -> dict | None:
        return self._market_metadata_by_ticker.get(ticker)

    def _cache_market_metadata(self, markets: list[dict]) -> None:
        for market in markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue
            self._market_metadata_by_ticker[ticker] = _market_metadata_from_response(market)

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{self._cfg.kalshi.rest_base}{path}"
        headers = self._auth_headers("GET", path)
        try:
            async with self.session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    log.warning("Kalshi rate limited on %s", path)
                    return None
                if resp.status != 200:
                    log.warning("Kalshi %s: HTTP %d", path, resp.status)
                    return None
                return await resp.json()
        except Exception:
            log.exception("Kalshi REST error on %s", path)
            return None

    async def discover_markets(
        self, series_ticker: str
    ) -> list[dict]:
        """Discover active markets for a series (e.g. KXHIGHMIA)."""
        markets = []
        cursor = None

        while True:
            params = {
                "series_ticker": series_ticker,
                "status": "open",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._get("/markets", params)
            if not data:
                break

            batch = data.get("markets", [])
            self._cache_market_metadata(batch)
            markets.extend(batch)

            cursor = data.get("cursor")
            if not cursor or len(batch) < 200:
                break

        return markets

    async def get_orderbook(self, ticker: str) -> dict | None:
        """Fetch orderbook for a single market."""
        return await self._get(f"/markets/{ticker}/orderbook")

    async def get_all_tickers(self) -> list[str]:
        """Discover all active KMIA temperature bracket tickers."""
        tickers = []
        for series in [
            self._cfg.station.kalshi_high_series,
            self._cfg.station.kalshi_low_series,
        ]:
            if not series:
                continue
            markets = await self.discover_markets(series)
            for market in markets:
                ticker = market.get("ticker", "")
                if ticker:
                    tickers.append(ticker)
        return tickers

    async def get_market_snapshots(
        self, tickers: list[str]
    ) -> list[MarketSnapshot]:
        """Fetch orderbooks for multiple tickers and return snapshots."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        snapshots = []

        for ticker in tickers:
            data = await self.get_orderbook(ticker)
            if not data:
                continue

            metadata = self.get_market_metadata(ticker) or _parse_ticker(ticker)

            # Extract best prices from orderbook
            ob = data.get("orderbook", data)
            yes_bids = ob.get("yes", [])
            no_bids = ob.get("no", [])

            best_yes_bid = max((p for p, _ in yes_bids), default=None) if yes_bids else None
            best_no_bid = max((p for p, _ in no_bids), default=None) if no_bids else None
            best_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
            best_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None

            snapshots.append(MarketSnapshot(
                ticker=ticker,
                event_ticker=metadata.get("event_ticker", ""),
                series_ticker=metadata.get("series_ticker", ""),
                market_type=metadata.get("market_type", ""),
                forecast_date=metadata.get("forecast_date"),
                floor_strike=metadata.get("floor_strike"),
                cap_strike=metadata.get("cap_strike"),
                best_yes_bid_cents=best_yes_bid,
                best_yes_ask_cents=best_yes_ask,
                best_no_bid_cents=best_no_bid,
                best_no_ask_cents=best_no_ask,
                last_price_cents=metadata.get("last_price_cents"),
                volume=metadata.get("volume"),
                snapshot_time=now_utc,
            ))

        return snapshots

    async def check_settlements(self) -> list[dict]:
        """Check for recently settled markets."""
        settled = []
        for series in [
            self._cfg.station.kalshi_high_series,
            self._cfg.station.kalshi_low_series,
        ]:
            if not series:
                continue
            params = {
                "series_ticker": series,
                "status": "settled",
                "limit": 50,
            }
            data = await self._get("/markets", params)
            if data:
                batch = data.get("markets", [])
                self._cache_market_metadata(batch)
                settled.extend(batch)
        return settled
