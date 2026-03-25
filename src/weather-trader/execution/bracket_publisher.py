"""Bracket publisher — discovers active Kalshi brackets and pushes definitions to city DBs.

The trader owns the Kalshi connection. When it discovers active markets,
it parses the bracket settlement ranges and writes them to each city's
`active_brackets` table so the city estimator can assign probabilities.

Also cleans up expired brackets (past dates).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class BracketDef:
    """Parsed bracket definition ready to write to city DB."""
    ticker: str
    market_type: str
    target_date: str
    floor_strike: float | None
    cap_strike: float | None
    settlement_floor: float
    settlement_ceil: float


def parse_bracket_range(ticker: str, floor_strike: float | None,
                        cap_strike: float | None) -> tuple[float, float] | None:
    """Convert Kalshi ticker + strikes into settlement °F range.

    Kalshi weather settles on whole degrees.
    - T57 (tail-low): "under 57" → settles ≤56°F → range (-∞, 57)
    - B57.5 (between): "57 or 58" → range [57, 59)
    - T64 (tail-high): "64 or above" → settles ≥63°F → range [63, +∞)
    """
    upper = ticker.upper()
    match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", upper)
    if not match:
        return None

    side = match.group(1)
    strike = float(match.group(2))

    if side == "T":
        # Tail bracket
        if "LOW" in upper:
            # For LOW markets:
            # T57 with floor=57, cap=56 → "under 57" → ≤56°F
            # T64 with floor=64 → "64 or above" → ≥63°F
            if cap_strike is not None and (floor_strike is None or cap_strike < floor_strike):
                # Tail-low: under the strike
                return -50.0, cap_strike + 1.0
            else:
                # Tail-high: above the value
                return strike - 1.0, 200.0
        else:
            # For HIGH markets:
            # T73 with floor=73, cap=72 → "under 73" → ≤72°F
            # T80 → "80 or above" → ≥79°F
            if cap_strike is not None and (floor_strike is None or cap_strike < floor_strike):
                return -50.0, cap_strike + 1.0
            else:
                return strike - 1.0, 200.0

    elif side == "B":
        # Between bracket: 2°F window
        # B57.5 → 57 or 58 wins → [57, 59)
        base = int(strike - 0.5) if strike != int(strike) else int(strike)
        return float(base), float(base + 2)

    return None


def discover_and_publish(
    kalshi_markets: list[dict],
    city_dbs: dict[str, sqlite3.Connection],
    target_date: str,
) -> int:
    """Parse Kalshi markets and write bracket definitions to city DBs.

    Args:
        kalshi_markets: Raw market dicts from Kalshi API (with ticker, floor_strike, etc.)
        city_dbs: Map of station → city DB connection
        target_date: Date to publish brackets for

    Returns:
        Number of brackets published.
    """
    brackets: list[BracketDef] = []

    for mkt in kalshi_markets:
        ticker = mkt.get("ticker", "")
        if not ticker:
            continue

        floor = mkt.get("floor_strike")
        cap = mkt.get("cap_strike")
        market_type = mkt.get("market_type", "")

        rng = parse_bracket_range(ticker, floor, cap)
        if rng is None:
            log.warning("Could not parse bracket range for %s", ticker)
            continue

        brackets.append(BracketDef(
            ticker=ticker,
            market_type=market_type,
            target_date=target_date,
            floor_strike=floor,
            cap_strike=cap,
            settlement_floor=rng[0],
            settlement_ceil=rng[1],
        ))

    # Write to all city DBs
    count = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%fZ")

    for station, db in city_dbs.items():
        for b in brackets:
            # Only write brackets that match this station's series
            # Station KMIA → look for "MIA" in ticker (e.g., KXHIGHMIA)
            station_code = station.upper().lstrip("K")  # KMIA → MIA
            if station_code not in b.ticker.upper():
                continue
            try:
                db.execute(
                    """INSERT OR REPLACE INTO active_brackets
                       (ticker, market_type, target_date, floor_strike, cap_strike,
                        settlement_floor, settlement_ceil, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (b.ticker, b.market_type, b.target_date,
                     b.floor_strike, b.cap_strike,
                     b.settlement_floor, b.settlement_ceil, now),
                )
                count += 1
            except Exception:
                log.exception("Failed to write bracket %s to %s", b.ticker, station)

        # Clean up expired brackets
        db.execute(
            "DELETE FROM active_brackets WHERE target_date < ?",
            (target_date,),
        )
        db.commit()

    log.info("Published %d brackets for %s across %d cities",
             count, target_date, len(city_dbs))
    return count
