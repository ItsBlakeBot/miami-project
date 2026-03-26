"""Main async orchestrator — 12 concurrent collection loops."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from collector.config import Config, load_config
from collector.sources.fawn import FAWNClient
from collector.sources.iem import IEMClient
from collector.sources.kalshi_rest import KalshiREST, _parse_ticker
from collector.sources.kalshi_ws import KalshiWS
from collector.sources.ndbc import NDBCClient
from collector.sources.nws import NWS
from collector.sources.openmeteo import OpenMeteo
from collector.sources.wethr import WethrREST
from collector.sources.wethr_stream import WethrStream
from analyzer.obs_analyzer import run_daily_obs_scoring
from collector.store.db import Store
from collector.types import EventSettlement, MarketSnapshot, Observation

log = logging.getLogger("collector")


# ---------------------------------------------------------------------------
# Shared live state (passed between loops to reduce latency)
# ---------------------------------------------------------------------------

class LiveState:
    """In-memory caches shared across async loops."""

    def __init__(self) -> None:
        # Obs-triggered inference event
        self.obs_changed = asyncio.Event()
        self._last_inference_temp: float | None = None
        self._last_inference_time: float = 0.0
        self.inference_cooldown_secs: float = 5.0

        # Active order book cache (updated by Kalshi WS on every tick)
        # Full book state per ticker for instant access — no DB round-trip needed.
        self.market_prices: dict[str, float] = {}  # ticker -> last_price_cents
        self.order_book: dict[str, dict] = {}  # ticker -> {best_yes_bid, best_yes_ask, best_no_bid, best_no_ask, last_price, updated_at}

        # Running high/low from wethr envelope (reset at climate day boundary)
        self.running_high_f: float | None = None
        self.running_low_f: float | None = None
        self._climate_day: str | None = None  # YYYY-MM-DD

    def update_obs(self, temperature_f: float | None, wethr_high_f: float | None, wethr_low_f: float | None, climate_day: str) -> None:
        """Called from wethr SSE loop on each observation."""
        import time as _time

        # Reset caches on climate day rollover
        if climate_day != self._climate_day:
            self.running_high_f = None
            self.running_low_f = None
            self._climate_day = climate_day

        # Update running extremes from wethr envelope (preferred) or raw temp
        high_val = wethr_high_f or temperature_f
        low_val = wethr_low_f or temperature_f
        if high_val is not None:
            if self.running_high_f is None or high_val > self.running_high_f:
                self.running_high_f = high_val
        if low_val is not None:
            if self.running_low_f is None or low_val < self.running_low_f:
                self.running_low_f = low_val

        self.signal_new_data()

    def mark_inference_ran(self, current_temp: float | None) -> None:
        import time as _time
        self._last_inference_temp = current_temp
        self._last_inference_time = _time.monotonic()

    def update_market_price(self, ticker: str, last_price_cents: float | None) -> None:
        if last_price_cents is not None:
            self.market_prices[ticker] = last_price_cents

    def update_order_book(
        self, ticker: str, *,
        best_yes_bid: float | None = None,
        best_yes_ask: float | None = None,
        best_no_bid: float | None = None,
        best_no_ask: float | None = None,
        last_price: float | None = None,
    ) -> None:
        """Update the full order book cache for a ticker. Called from Kalshi WS."""
        import time as _time
        book = self.order_book.get(ticker, {})
        if best_yes_bid is not None:
            book["best_yes_bid"] = best_yes_bid
        if best_yes_ask is not None:
            book["best_yes_ask"] = best_yes_ask
        if best_no_bid is not None:
            book["best_no_bid"] = best_no_bid
        if best_no_ask is not None:
            book["best_no_ask"] = best_no_ask
        if last_price is not None:
            book["last_price"] = last_price
        book["updated_at"] = _time.monotonic()
        self.order_book[ticker] = book

    def signal_new_data(self) -> None:
        """Signal that new data arrived from any source. Triggers inference
        if cooldown has elapsed."""
        import time as _time
        elapsed = _time.monotonic() - self._last_inference_time
        if elapsed >= self.inference_cooldown_secs:
            self.obs_changed.set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_dsm_summary(station: str, payload: dict) -> None:
    """Log DSM daily summary values."""
    for_date = payload.get("for_date", "?")
    high_f = payload.get("high_f")
    low_f = payload.get("low_f")
    log.info("DSM summary: %s high=%s low=%s", for_date, high_f, low_f)


def _ingest_dsm_settlement(station: str, payload: dict, store: Store) -> None:
    """Parse DSM event and write high/low into event_settlements.

    DSM provides the NWS daily summary (same basis Kalshi uses).
    Each new DSM for the same date overwrites previous values via REPLACE.
    """
    import json

    for_date = payload.get("for_date")
    if not for_date:
        return

    received = payload.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    raw_text = json.dumps(payload)

    high_f = payload.get("high_f")
    if high_f is not None:
        try:
            store.insert_event_settlement(EventSettlement(
                station=station,
                settlement_date=for_date,
                market_type="high",
                actual_value_f=float(high_f),
                settlement_source="dsm",
                raw_text=raw_text,
                received_at=received,
            ))
            log.info("DSM settlement: %s high=%s", for_date, high_f)
        except Exception:
            log.exception("Failed to insert DSM high settlement")

    low_f = payload.get("low_f")
    if low_f is not None:
        try:
            store.insert_event_settlement(EventSettlement(
                station=station,
                settlement_date=for_date,
                market_type="low",
                actual_value_f=float(low_f),
                settlement_source="dsm",
                raw_text=raw_text,
                received_at=received,
            ))
            log.info("DSM settlement: %s low=%s", for_date, low_f)
        except Exception:
            log.exception("Failed to insert DSM low settlement")


def _lst_today(utc_offset: int) -> str:
    """Current LST date as YYYY-MM-DD."""
    now = datetime.now(timezone.utc) + timedelta(hours=utc_offset)
    return now.strftime("%Y-%m-%d")


def _lst_tomorrow(utc_offset: int) -> str:
    now = datetime.now(timezone.utc) + timedelta(hours=utc_offset) + timedelta(days=1)
    return now.strftime("%Y-%m-%d")


def _upsert_active_bracket(store: Store, ticker: str, parsed: dict) -> None:
    """Write a discovered bracket to active_brackets from WS/REST metadata."""
    forecast_date = parsed.get("forecast_date")
    market_type = parsed.get("market_type", "")
    floor_strike = parsed.get("floor_strike")
    cap_strike = parsed.get("cap_strike")

    if not forecast_date or not market_type:
        return

    # Continuous settlement bounds with half-degree continuity correction.
    # Kalshi settles on whole-degree F (NWS rounding).
    # B77.5 wins on 77 or 78 → true temp in [76.5, 78.5).
    # T73 under wins on ≤72 (<73) → true temp < 72.5.
    # T80 over wins on ≥81 (>80) → true temp ≥ 80.5.
    parts = ticker.split("-")
    strike_part = parts[-1] if len(parts) >= 3 else ""

    if strike_part.startswith("T"):
        f = floor_strike or 0.0
        if cap_strike is not None and floor_strike is not None and cap_strike < floor_strike:
            # Under: T73 → wins on ≤72 → true temp < 72.5
            settlement_floor, settlement_ceil = -50.0, f - 0.5
        else:
            # Over: T80 → wins on ≥81 (>80) → true temp ≥ 80.5
            settlement_floor, settlement_ceil = f + 0.5, 200.0
    elif strike_part.startswith("B"):
        f = floor_strike or 0.0
        settlement_floor, settlement_ceil = f - 0.5, f + 1.5
    else:
        settlement_floor = floor_strike or 0.0
        settlement_ceil = cap_strike or 200.0

    try:
        store.conn.execute(
            """INSERT OR REPLACE INTO active_brackets
               (ticker, market_type, target_date, floor_strike, cap_strike,
                settlement_floor, settlement_ceil)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ticker, market_type, forecast_date, floor_strike, cap_strike,
             settlement_floor, settlement_ceil),
        )
        store.conn.commit()
    except Exception:
        log.exception("Failed to upsert active bracket %s", ticker)


# ---------------------------------------------------------------------------
# Loop 1: Wethr SSE (always-on, 1-min obs)
# ---------------------------------------------------------------------------

async def wethr_sse_loop(cfg: Config, store: Store, live: LiveState | None = None) -> None:
    obs_count = 0
    offset = cfg.station.utc_offset_hours

    def on_observation(obs: Observation) -> None:
        nonlocal obs_count
        try:
            store.insert_observation(obs)
            obs_count += 1
            if obs_count % 10 == 1:
                log.info("WethrSSE obs #%d: %.1fF at %s",
                         obs_count, obs.temperature_f or 0, obs.timestamp_utc)
            # Update live state caches
            if live is not None:
                climate_day = _lst_today(offset)
                live.update_obs(
                    temperature_f=obs.temperature_f,
                    wethr_high_f=getattr(obs, "wethr_high_f", None),
                    wethr_low_f=getattr(obs, "wethr_low_f", None),
                    climate_day=climate_day,
                )
        except Exception:
            log.exception("Failed to store SSE observation")

    def on_event(station: str, event_type: str, payload: dict) -> None:
        try:
            store.insert_sse_event(station, event_type, payload)
            log.info("WethrSSE event: %s", event_type)

            # DSM provides running daily high/low — log for reference but
            # DO NOT write to event_settlements. DSM values are preliminary
            # (the climate day may still be open). Only the FINAL CLI after
            # the climate day ends should populate event_settlements.
            # The Wethr envelope (running high/low in LiveState) already
            # tracks the current extremes in real-time.
            if event_type == "dsm":
                _log_dsm_summary(station, payload)
                # _ingest_dsm_settlement DISABLED — use final CLI only
                # DSM data is logged in sse_events table for research
        except Exception:
            log.exception("Failed to store SSE event")

    stream = WethrStream(cfg, on_observation=on_observation, on_event=on_event)
    await stream.start()


# ---------------------------------------------------------------------------
# Loop 2: Kalshi WebSocket (always-on, orderbook deltas)
# ---------------------------------------------------------------------------

async def kalshi_ws_loop(
    cfg: Config, store: Store, kalshi_ws: KalshiWS, kalshi_rest: KalshiREST,
    live: LiveState | None = None,
) -> None:
    now_utc_fn = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Track which tickers we've already written to active_brackets this session
    _brackets_written: set[str] = set()

    def on_snapshot(ticker: str, prices: dict) -> None:
        try:
            metadata = kalshi_rest.get_market_metadata(ticker)
            parsed = metadata or _parse_ticker(ticker)

            # Write market snapshot (orderbook data)
            store.insert_market_snapshot(MarketSnapshot(
                ticker=ticker,
                event_ticker=parsed.get("event_ticker", ""),
                series_ticker=parsed.get("series_ticker", ""),
                market_type=parsed.get("market_type", ""),
                forecast_date=parsed.get("forecast_date"),
                floor_strike=parsed.get("floor_strike"),
                cap_strike=parsed.get("cap_strike"),
                best_yes_bid_cents=prices.get("best_yes_bid_cents"),
                best_yes_ask_cents=prices.get("best_yes_ask_cents"),
                best_no_bid_cents=prices.get("best_no_bid_cents"),
                best_no_ask_cents=prices.get("best_no_ask_cents"),
                last_price_cents=prices.get("last_price_cents"),
                volume=prices.get("volume"),
                yes_bid_qty=prices.get("yes_bid_qty"),
                yes_ask_qty=prices.get("yes_ask_qty"),
                no_bid_qty=prices.get("no_bid_qty"),
                no_ask_qty=prices.get("no_ask_qty"),
                total_yes_depth=prices.get("total_yes_depth"),
                total_no_depth=prices.get("total_no_depth"),
                spread_cents=prices.get("spread_cents"),
                num_yes_levels=prices.get("num_yes_levels"),
                num_no_levels=prices.get("num_no_levels"),
                snapshot_time=now_utc_fn(),
            ))

            # Update live price + full order book cache and signal new data
            if live is not None:
                live.update_market_price(ticker, prices.get("last_price_cents"))
                live.update_order_book(
                    ticker,
                    best_yes_bid=prices.get("best_yes_bid_cents"),
                    best_yes_ask=prices.get("best_yes_ask_cents"),
                    best_no_bid=prices.get("best_no_bid_cents"),
                    best_no_ask=prices.get("best_no_ask_cents"),
                    last_price=prices.get("last_price_cents"),
                )
                live.signal_new_data()

            # Also populate active_brackets (once per ticker per session)
            if ticker not in _brackets_written:
                _upsert_active_bracket(store, ticker, parsed)
                _brackets_written.add(ticker)

        except Exception:
            log.exception("Failed to store WS market snapshot")

    kalshi_ws.set_snapshot_callback(on_snapshot)
    await kalshi_ws.start()


# ---------------------------------------------------------------------------
# Loop 3: Forecast polling (Wethr REST + Open-Meteo, 5 min)
# ---------------------------------------------------------------------------

async def forecast_loop(
    cfg: Config, store: Store, wethr: WethrREST, openmeteo: OpenMeteo
) -> None:
    station = cfg.station.code
    offset = cfg.station.utc_offset_hours
    interval = cfg.wethr.forecast_poll_secs

    while True:
        try:
            today = _lst_today(offset)
            tomorrow = _lst_tomorrow(offset)
            total = 0

            for date_str in [today, tomorrow]:
                # Wethr NWP forecasts
                forecasts = await wethr.get_forecasts(station, date_str)
                total += store.insert_forecasts(forecasts)

                # Wethr NWS forecasts
                nws_fcsts = await wethr.get_nws_forecasts(station, date_str)
                total += store.insert_forecasts(nws_fcsts)

                # Open-Meteo deterministic
                om_det = await openmeteo.get_deterministic(station, date_str)
                total += store.insert_forecasts(om_det)

                # Open-Meteo ECMWF-AIFS (separate endpoint)
                om_aifs = await openmeteo.get_ecmwf_aifs(station, date_str)
                total += store.insert_forecasts(om_aifs)

                # Open-Meteo ensemble
                om_ens = await openmeteo.get_ensemble(station, date_str)
                total += store.insert_forecasts(om_ens)

            store.log_collection_run("forecasts", "ok", total)
            log.info("Forecast loop: %d records", total)
            if total > 0:
                store.signal_new_data()
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Forecast loop error")
            store.log_collection_run("forecasts", "error", 0)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Loop 4: NWS polling (obs 2.5 min, CLI 30 min)
# ---------------------------------------------------------------------------

async def nws_loop(cfg: Config, store: Store, nws: NWS) -> None:
    station = cfg.station.code
    obs_interval = cfg.nws.obs_poll_secs
    cli_interval = cfg.nws.cli_poll_secs
    cli_location = cfg.station.nws_cli_location

    last_cli_check = 0.0

    while True:
        try:
            # Observation
            obs = await nws.get_latest_observation(station)
            if obs:
                store.insert_observation(obs)

            # CLI check (less frequent)
            # CRITICAL: Only accept CLI settlements for climate days that have
            # ENDED. Climate day runs 05:00Z to 04:59Z next day. A CLI for
            # date D is only final if we're past 05:00Z on D+1.
            # Premature CLI/DSM data should NOT overwrite running extremes.
            import time
            now_mono = time.monotonic()
            if now_mono - last_cli_check >= cli_interval:
                settlements = await nws.get_cli_report(cli_location, station)
                now_utc_dt = datetime.now(timezone.utc)
                utc_offset = cfg.station.utc_offset_hours  # -5 for KMIA
                for s in settlements:
                    # Climate day D ends at 05:00Z on D+1.
                    # Only accept if now > settlement_date + 1 day + 05:00Z
                    try:
                        sdate = datetime.strptime(s.settlement_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        climate_day_end = sdate + timedelta(days=1, hours=5)  # 05:00Z next day
                        if now_utc_dt >= climate_day_end:
                            store.insert_event_settlement(s)
                            log.info("CLI settlement: %s %s=%s", s.settlement_date, s.market_type, s.actual_value_f)
                        else:
                            log.debug(
                                "CLI settlement BLOCKED (climate day still open): %s %s=%s (ends %s)",
                                s.settlement_date, s.market_type, s.actual_value_f,
                                climate_day_end.strftime("%Y-%m-%dT%H:%MZ"),
                            )
                    except (ValueError, TypeError):
                        # Can't parse date — skip rather than write bad data
                        log.warning("CLI settlement skipped (bad date): %s", s.settlement_date)
                last_cli_check = now_mono

            store.log_collection_run("nws", "ok", 1)
            store.signal_new_data()
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("NWS loop error")
            store.log_collection_run("nws", "error", 0)

        await asyncio.sleep(obs_interval)


# ---------------------------------------------------------------------------
# Loop 5: Kalshi REST (market discovery 10 min, settlement checks)
# ---------------------------------------------------------------------------

async def kalshi_rest_loop(
    cfg: Config, store: Store, kalshi_rest: KalshiREST, kalshi_ws: KalshiWS
) -> None:
    interval = cfg.kalshi.market_discovery_secs

    while True:
        try:
            # Discover active tickers and populate active_brackets
            tickers = await kalshi_rest.get_all_tickers()
            log.info("Kalshi discovery: %d active tickers", len(tickers))

            for ticker in tickers:
                parsed = kalshi_rest.get_market_metadata(ticker) or _parse_ticker(ticker)
                _upsert_active_bracket(store, ticker, parsed)

            # Update WS subscriptions
            kalshi_ws.set_tickers(set(tickers))
            if kalshi_ws.is_connected:
                await kalshi_ws.sync_subscriptions()

            # Check settlements
            settled = await kalshi_rest.check_settlements()
            for m in settled:
                ticker = m.get("ticker", "")
                result = m.get("result", "")
                if ticker and result:
                    parsed = kalshi_rest.get_market_metadata(ticker) or _parse_ticker(ticker)

                    store.insert_market_settlement(
                        ticker=ticker,
                        event_ticker=parsed.get("event_ticker", ""),
                        series_ticker=parsed.get("series_ticker", ""),
                        station=cfg.station.code,
                        forecast_date=parsed.get("forecast_date") or m.get("expiration_time", "")[:10],
                        market_type=parsed.get("market_type", ""),
                        floor_strike=parsed.get("floor_strike"),
                        cap_strike=parsed.get("cap_strike"),
                        winning_side=result,
                        settled_at=m.get("close_time", ""),
                    )

            store.log_collection_run("kalshi_rest", "ok", len(tickers))
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Kalshi REST loop error")
            store.log_collection_run("kalshi_rest", "error", 0)

        await asyncio.sleep(interval)


# (Loop 6 — pressure_level_loop — REMOVED: data never used in inference)
# (Loop 7 — synoptic_loop — REMOVED: API permanently down, replaced by IEM)


# ---------------------------------------------------------------------------
# Loop 8: IEM nearby station polling (5 min, no auth)
# ---------------------------------------------------------------------------

async def iem_loop(cfg: Config, store: Store) -> None:
    """Poll nearby FL ASOS stations via IEM for spatial temperature/wind data."""
    client = IEMClient(cfg, radius_mi=cfg.iem.radius_mi)
    await client.open()
    interval = cfg.iem.poll_secs
    offset = cfg.station.utc_offset_hours

    try:
        while True:
            try:
                nearby = await client.get_latest_nearby()

                if not nearby:
                    store.log_collection_run("iem", "ok", 0)
                    await asyncio.sleep(interval)
                    continue

                kmia_temp = store.get_latest_kmia_temp()

                now = datetime.now(timezone.utc) + timedelta(hours=offset)
                lst_date = now.strftime("%Y-%m-%d")

                batch = []
                for obs in nearby:
                    delta = None
                    if kmia_temp is not None and obs.air_temp_f is not None:
                        delta = round(obs.air_temp_f - kmia_temp, 1)

                    batch.append((
                        obs.icao, obs.name, obs.network,
                        obs.latitude, obs.longitude, obs.distance_mi,
                        obs.elevation_m, obs.timestamp_utc, lst_date,
                        obs.air_temp_f, obs.dew_point_f,
                        obs.wind_speed_mph, obs.wind_direction_deg,
                        obs.wind_gust_mph, obs.pressure_slp_hpa,
                        obs.sky_cover_code, delta,
                    ))

                    store.upsert_nearby_station(
                        obs.icao, obs.name, obs.network,
                        obs.latitude, obs.longitude, obs.distance_mi,
                        obs.elevation_m, obs.bearing_deg,
                    )

                count = store.insert_nearby_observations(batch)
                store.log_collection_run("iem", "ok", count)
                if count > 0:
                    store.signal_new_data()
                log.info("IEM nearby: %d new obs from %d stations, KMIA temp=%.1f",
                         count, len(batch), kmia_temp or 0)

            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("IEM loop error")
                store.log_collection_run("iem", "error", 0)

            await asyncio.sleep(interval)
    finally:
        await client.close()


# (Loop 9 — atmospheric_loop — REMOVED: CAPE/PW/radiation migrated to HRRR/GOES/FAWN;
#  remaining precip fields never used in inference)


# ---------------------------------------------------------------------------
# Loop 10: NDBC SST polling (30 min)
# ---------------------------------------------------------------------------

async def ndbc_loop(cfg: Config, store: Store) -> None:
    """Poll NDBC buoys for sea surface temperature."""
    client = NDBCClient(cfg)
    await client.open()
    interval = cfg.ndbc.poll_secs

    try:
        while True:
            try:
                obs = await client.get_latest_sst()
                count = store.insert_sst_observations(obs)
                store.log_collection_run("ndbc_sst", "ok", count)
                if count > 0:
                    store.signal_new_data()
                if count:
                    log.info("NDBC SST: %d new observations", count)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("NDBC loop error")
                store.log_collection_run("ndbc_sst", "error", 0)

            await asyncio.sleep(interval)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Loop 11: FAWN actual sensor data (15 min)
# ---------------------------------------------------------------------------

async def fawn_loop(cfg: Config, store: Store) -> None:
    """Poll FAWN stations for actual solar radiation, precip, soil temp.

    Fetches from all configured station_ids (default: just Homestead).
    Multi-station support enables spatial coverage for LETKF clusters.
    """
    station_ids = list(cfg.fawn.station_ids) if cfg.fawn.station_ids else [cfg.fawn.station_id]
    client = FAWNClient(cfg, station_ids=station_ids)
    await client.open()
    interval = cfg.fawn.poll_secs

    try:
        while True:
            try:
                if len(station_ids) > 1:
                    obs = await client.get_all_stations_obs()
                else:
                    obs = await client.get_today_obs()
                count = store.insert_fawn_observations(obs)
                store.log_collection_run("fawn", "ok", count)
                if count > 0:
                    store.signal_new_data()
                if count:
                    log.info("FAWN: %d new observations from %d stations", count, len(station_ids))
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("FAWN loop error")
                store.log_collection_run("fawn", "error", 0)

            await asyncio.sleep(interval)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Loop 12a: HRRR atmospheric data via Herbie (hourly — replaces Open-Meteo atmos)
# ---------------------------------------------------------------------------

async def hrrr_atmospheric_loop(cfg: Config, store: Store) -> None:
    """Poll HRRR for CAPE, PW, cloud cover, radiation, and upper-air data.

    Runs hourly (matching HRRR cycle). Writes to atmospheric_data table
    with model='HRRR', providing free high-resolution atmospheric fields
    that can supplement or replace Open-Meteo atmospheric data.
    """
    from collector.sources.hrrr import fetch_hrrr_atmospheric
    from collector.types import AtmosphericData
    from datetime import timedelta

    interval = 3600  # Every hour — matches HRRR update cycle
    lat = cfg.station.lat
    lon = cfg.station.lon
    station = cfg.station.code

    while True:
        try:
            obs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fetch_hrrr_atmospheric(lat, lon, station=station),
            )
            if obs is not None:
                atmos = AtmosphericData(
                    station=station,
                    valid_time_utc=obs.valid_time_utc,
                    model="HRRR",
                    fetch_time_utc=_now_utc(),
                    shortwave_radiation=obs.dswrf_wm2,
                    cape=obs.cape_jkg,
                    boundary_layer_height=obs.hpbl_m,
                    precipitable_water_mm=obs.pwat_mm,
                )
                count = store.insert_atmospheric_data(atmos)
                store.log_collection_run("hrrr_atmos", "ok", count)
                if count > 0:
                    store.signal_new_data()
                    log.info(
                        "HRRR atmos: CAPE=%.0f PW=%.1f cloud=%.0f%% DSWRF=%.0f",
                        obs.cape_jkg or 0, obs.pwat_mm or 0,
                        obs.cloud_cover_pct or 0, obs.dswrf_wm2 or 0,
                    )
            else:
                store.log_collection_run("hrrr_atmos", "no_data", 0)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("HRRR atmospheric loop error")
            store.log_collection_run("hrrr_atmos", "error", 0)

        await asyncio.sleep(interval)


def _now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Loop 12c: RTMA-RU (Real-Time Mesoscale Analysis - Rapid Update, every 15 min)
# ---------------------------------------------------------------------------

async def rtma_ru_loop(cfg: Config, store: Store) -> None:
    """Poll RTMA-RU for analyzed surface fields every 15 minutes.

    RTMA-RU assimilates all real-time observations into a 2.5km grid,
    providing the best available "what IS the temperature right now" product.
    15-minute update cycle, ~20 minute latency.

    Writes to rtma_ru_observations table (center + 5x5 spatial grid).
    """
    from collector.sources.rtma_ru import fetch_rtma_ru

    interval = 900  # 15 minutes
    lat = cfg.station.lat
    lon = cfg.station.lon
    station = cfg.station.code

    # Align to next :00/:15/:30/:45 boundary + latency buffer
    while True:
        try:
            obs_list = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fetch_rtma_ru(lat, lon),
            )
            if obs_list:
                count = store.insert_rtma_ru_observations(obs_list)
                store.log_collection_run("rtma_ru", "ok", count)
                if count > 0:
                    store.signal_new_data()
                    # Find center point for logging
                    center = min(
                        obs_list,
                        key=lambda o: (o.lat - lat) ** 2 + (o.lon - lon) ** 2,
                    )
                    temp_f = center.temperature_2m * 9 / 5 + 32 if center.temperature_2m is not None else None
                    log.info(
                        "RTMA-RU: %s T=%.1f°F DP=%.1f°C WS=%.1f m/s @ %.0f° P=%.1f hPa (%d pts)",
                        center.timestamp_utc,
                        temp_f or 0,
                        center.dewpoint_2m or 0,
                        center.wind_speed_10m or 0,
                        center.wind_direction_10m or 0,
                        center.surface_pressure or 0,
                        count,
                    )
            else:
                store.log_collection_run("rtma_ru", "no_data", 0)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("RTMA-RU loop error")
            store.log_collection_run("rtma_ru", "error", 0)

        # Sleep until next 15-min boundary + 35 min latency buffer
        now = datetime.now(timezone.utc)
        next_quarter = now.replace(second=0, microsecond=0)
        next_quarter += timedelta(minutes=(15 - now.minute % 15))
        # Add ~35 min for RTMA-RU production latency
        sleep_secs = max(60, (next_quarter - now).total_seconds())
        await asyncio.sleep(sleep_secs)


# ---------------------------------------------------------------------------
# Loop 12b: GOES-19 satellite observations (every 10 min)
# ---------------------------------------------------------------------------

async def goes_satellite_loop(cfg: Config, store: Store) -> None:
    """Poll GOES-19 for live satellite-derived cloud cover, CAPE, and stability.

    Provides ACTUAL OBSERVATIONS from satellite, not model forecasts.
    - Clear Sky Mask: cloud/clear classification, 2km, 5-min CONUS
    - Derived Stability Index: CAPE, Lifted Index, K-Index (clear-sky only)

    Stores to atmospheric_data table with model='GOES-19-SAT' to distinguish
    from model-derived atmospheric data (HRRR, Open-Meteo).
    """
    from collector.sources.goes import fetch_goes_observations
    from collector.types import AtmosphericData

    interval = 600  # Every 10 min (GOES scans every 5 min, but we don't need that fast)
    lat = cfg.station.lat
    lon = cfg.station.lon
    station = cfg.station.code

    while True:
        try:
            obs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: fetch_goes_observations(lat, lon, station=station),
            )
            if obs is not None:
                atmos = AtmosphericData(
                    station=station,
                    valid_time_utc=obs.timestamp_utc,
                    model="GOES-19-SAT",
                    fetch_time_utc=_now_utc(),
                    shortwave_radiation=obs.dswrf_wm2,
                    cape=obs.cape_jkg,
                    lifted_index=obs.lifted_index,
                )
                count = store.insert_atmospheric_data(atmos)
                store.log_collection_run("goes_satellite", "ok", count)
                if count > 0:
                    store.signal_new_data()
                    log.info(
                        "GOES-19: cloud=%.2f CAPE=%.0f LI=%.1f",
                        obs.cloud_fraction or -1,
                        obs.cape_jkg or 0,
                        obs.lifted_index or 0,
                    )
            else:
                store.log_collection_run("goes_satellite", "no_data", 0)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("GOES satellite loop error")
            store.log_collection_run("goes_satellite", "error", 0)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Loop 12: NWS gridpoint data polling (hourly)
# ---------------------------------------------------------------------------

async def nws_gridpoint_loop(cfg: Config, store: Store, nws: NWS) -> None:
    """Poll NWS gridpoint for mixing height and transport wind."""
    interval = 1800  # Every 30 min — NWS gridpoint updates with each model run

    while True:
        try:
            data = await nws.get_gridpoint_mixing_height()
            if data:
                store.insert_sse_event(
                    cfg.station.code, "nws_gridpoint",
                    data,
                )
                total = sum(len(v) for v in data.values())
                store.log_collection_run("nws_gridpoint", "ok", total)
                log.info("NWS gridpoint: %d data points", total)
                if total > 0:
                    store.signal_new_data()
            else:
                store.log_collection_run("nws_gridpoint", "ok", 0)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("NWS gridpoint loop error")
            store.log_collection_run("nws_gridpoint", "error", 0)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Loop 12d: Herbie NWP forecast polling (GFS, ECMWF IFS, NBM — every 90 min)
# ---------------------------------------------------------------------------

async def herbie_nwp_loop(cfg: Config, store: Store) -> None:
    """Poll free NWP model forecasts via Herbie for GFS, ECMWF IFS, and NBM.

    These replace the Open-Meteo paid API for the three highest-value models.
    Runs every 90 minutes — models update every 6h but we check more frequently
    to catch new runs as they become available.
    """
    from collector.sources.herbie_nwp import (
        fetch_gfs_forecasts,
        fetch_ecmwf_ifs_forecasts,
        fetch_nbm_forecasts,
    )

    interval = 5400  # 90 minutes
    lat = cfg.station.lat
    lon = cfg.station.lon
    station = cfg.station.code
    offset = cfg.station.utc_offset_hours

    while True:
        now = datetime.now(timezone.utc) + timedelta(hours=offset)
        target_date = now.strftime("%Y-%m-%d")
        total = 0

        for name, fetcher in [
            ("GFS-Herbie", fetch_gfs_forecasts),
            ("ECMWF-IFS-Herbie", fetch_ecmwf_ifs_forecasts),
            ("NBM-Herbie", fetch_nbm_forecasts),
        ]:
            try:
                fcsts = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda f=fetcher: f(lat, lon, station, target_date),
                )
                count = store.insert_forecasts(fcsts)
                total += count
                store.log_collection_run(f"herbie_{name}", "ok", count)
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception("Herbie %s fetch error", name)
                store.log_collection_run(f"herbie_{name}", "error", 0)

        if total > 0:
            store.signal_new_data()
            log.info("Herbie NWP: %d total forecasts for %s", total, target_date)

        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Loop 13: Daily scoring pipeline
# ---------------------------------------------------------------------------

async def scoring_loop(cfg: Config, store: Store) -> None:
    """Run observation scoring once daily after CLI arrives (~11:00Z for Miami).

    Model scoring and bias adjustment are handled by model_scorer.py
    via the bias_adjust_loop (runs every 30 min).
    Obs scoring (cloud impact, etc.) runs once daily after settlement.
    """
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            dsm_hour = int(cfg.station.dsm_zulu.split(":")[0])
            if now_utc.hour >= dsm_hour + 1:
                today_lst = _lst_today(cfg.station.utc_offset_hours)
                yesterday = (
                    datetime.strptime(today_lst, "%Y-%m-%d") - timedelta(days=1)
                ).strftime("%Y-%m-%d")
                has_cli = store.conn.execute(
                    "SELECT 1 FROM event_settlements WHERE station=? AND settlement_date=? AND settlement_source='cli' LIMIT 1",
                    (cfg.station.code, yesterday),
                ).fetchone()
                if has_cli:
                    run_daily_obs_scoring(store, cfg.station.code, yesterday, cfg.station.utc_offset_hours)
                    log.info("Obs scoring complete for %s", yesterday)
                await asyncio.sleep(3600 * 20)
            else:
                await asyncio.sleep(1800)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Scoring loop error")
            await asyncio.sleep(3600)


async def bias_adjust_loop(cfg: Config, store: Store) -> None:
    """Recompute MAE-weighted consensus + save forward curves every 30 minutes."""
    from analyzer.model_scorer import run_bias_adjustment
    from collector.forward_curve import build_forward_curve

    while True:
        try:
            today = _lst_today(cfg.station.utc_offset_hours)
            tomorrow = (
                datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")

            # 1. Bias adjustment
            for date_str in [today, tomorrow]:
                high_r, low_r = run_bias_adjustment(
                    store._path, cfg.station.code, date_str
                )
                if high_r:
                    log.info("Bias-adj %s HIGH: %.1f°F (%d models)",
                             date_str, high_r.consensus_f, high_r.n_models)
                if low_r:
                    log.info("Bias-adj %s LOW: %.1f°F (%d models)",
                             date_str, low_r.consensus_f, low_r.n_models)

            # 2. Save forward curves for historical analysis
            try:
                now_utc = datetime.now(timezone.utc)
                fwd_rows = build_forward_curve(
                    store._path, cfg.station.code,
                    cfg.station.utc_offset_hours, today,
                    now_utc=now_utc,
                )
                if fwd_rows:
                    timestamp_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                    n = store.save_forward_curve(
                        cfg.station.code, timestamp_utc, today, fwd_rows
                    )
                    log.info("Saved %d forward curve rows for %s", n, today)
            except Exception:
                log.exception("Forward curve save error")

        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Bias adjust loop error")
        await asyncio.sleep(1800)  # 30 minutes


# ---------------------------------------------------------------------------
# Loop 15: Inference (5 min — runs full pipeline, writes bracket_estimates)
# ---------------------------------------------------------------------------

async def inference_loop(cfg: Config, store: Store, live: LiveState | None = None) -> None:
    from engine.orchestrator import InferenceRuntimeState, run_inference_cycle

    interval = 300  # 5 minutes
    runtime_state = InferenceRuntimeState()

    # Initialize LETKF for spatial assimilation (SE Florida cluster)
    try:
        from engine.letkf import SE_FLORIDA_CLUSTER, LETKFState
        letkf = LETKFState(cluster=SE_FLORIDA_CLUSTER)
        # Initialize with a generic forecast — will converge quickly with obs
        letkf.initialize_from_forecasts(
            {s.code: 75.0 for s in SE_FLORIDA_CLUSTER.stations},
            forecast_spread=3.0,
        )
        runtime_state.letkf_state = letkf
        log.info("LETKF initialized for %s cluster (%d stations)",
                 SE_FLORIDA_CLUSTER.name, SE_FLORIDA_CLUSTER.n_stations)
    except Exception:
        log.warning("LETKF initialization failed — spatial assimilation disabled", exc_info=True)

    # Wait for initial data to accumulate before first run
    await asyncio.sleep(30)

    while True:
        try:
            # Pass live market prices if available
            market_price_override = live.market_prices if live else None
            result = run_inference_cycle(
                store._path,
                market_prices_override=market_price_override,
                runtime_state=runtime_state,
            )
            n_estimates = len(result.estimates)
            log.info(
                "Inference: %s | %d estimates | %s",
                result.target_date, n_estimates,
                " | ".join(result.notes[-2:]) if result.notes else "ok",
            )
            for est in result.estimates:
                if est.edge is not None and abs(est.edge) >= 5.0:
                    log.info(
                        "  EDGE: %s %+.1f¢ (model=%.0f%%, mkt=%s¢)",
                        est.ticker, est.edge,
                        est.model_probability * 100,
                        est.market_price,
                    )

            store.log_collection_run("inference", "ok", n_estimates)

            # Run paper trader after every inference cycle
            try:
                from trading.paper_trader import PaperTrader
                trader = PaperTrader(
                    store._path,
                    station=cfg.station.code,
                    max_quote_age_minutes=20,
                )
                pt_result = trader.run_cycle()
                if pt_result.get("entries") or pt_result.get("exits"):
                    log.info(
                        "PaperTrader: %d entries, %d exits | adaptive:%s",
                        pt_result.get("entries", 0),
                        pt_result.get("exits", 0),
                        pt_result.get("adaptive", {}).get("mode", "unknown"),
                    )
            except Exception:
                log.exception("Paper trader error")

            # Mark inference ran for cooldown tracking
            if live is not None:
                # Use mu from first high estimate as representative temp
                current_temp = None
                for est in result.estimates:
                    if est.market_type == "high" and est.mu is not None:
                        current_temp = est.mu
                        break
                live.mark_inference_ran(current_temp)

        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Inference loop error")
            store.log_collection_run("inference", "error", 0)

        # Wait for either obs change event or timer expiry
        if live is not None:
            try:
                await asyncio.wait_for(live.obs_changed.wait(), timeout=interval)
                live.obs_changed.clear()
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# DS3M Shadow Loop (wrapper with lazy import)
# ---------------------------------------------------------------------------

async def _ds3m_shadow_wrapper(cfg: Config, store: Store, live: LiveState | None = None) -> None:
    """Thin wrapper for DS3M shadow loop with lazy import.

    DS3M runs in shadow mode — predicts bracket probabilities in parallel
    with production but never affects live trading. Writes to ds3m_* tables.
    """
    try:
        from engine.ds3m.orchestrator import ds3m_shadow_loop
        await ds3m_shadow_loop(cfg, store, live)
    except ImportError:
        log.warning("DS3M package not available; shadow loop disabled")
    except Exception:
        log.exception("DS3M shadow loop failed fatally")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(config_path: str = "config.toml") -> None:
    cfg = load_config(config_path)
    store = Store("miami_collector.db")
    store.open()

    # Initialize API clients
    wethr = WethrREST(cfg)
    openmeteo = OpenMeteo(cfg)
    nws = NWS(cfg)
    kalshi_rest = KalshiREST(cfg)
    kalshi_ws = KalshiWS(cfg)

    await asyncio.gather(
        wethr.open(), openmeteo.open(), nws.open(), kalshi_rest.open()
    )

    log.info("All clients initialized. Starting 17 loops (16 collection + DS3M shadow).")

    live = LiveState()
    # Attach to store so any loop can signal new data without
    # needing the live object passed explicitly.
    store._live = live  # type: ignore[attr-defined]

    # Launch concurrent loops
    tasks = [
        asyncio.create_task(wethr_sse_loop(cfg, store, live), name="wethr_sse"),
        asyncio.create_task(kalshi_ws_loop(cfg, store, kalshi_ws, kalshi_rest, live), name="kalshi_ws"),
        asyncio.create_task(forecast_loop(cfg, store, wethr, openmeteo), name="forecasts"),
        asyncio.create_task(nws_loop(cfg, store, nws), name="nws"),
        asyncio.create_task(kalshi_rest_loop(cfg, store, kalshi_rest, kalshi_ws), name="kalshi_rest"),
        asyncio.create_task(iem_loop(cfg, store), name="iem"),
        asyncio.create_task(ndbc_loop(cfg, store), name="ndbc_sst"),
        asyncio.create_task(fawn_loop(cfg, store), name="fawn"),
        asyncio.create_task(hrrr_atmospheric_loop(cfg, store), name="hrrr_atmos"),
        asyncio.create_task(goes_satellite_loop(cfg, store), name="goes_satellite"),
        asyncio.create_task(rtma_ru_loop(cfg, store), name="rtma_ru"),
        asyncio.create_task(herbie_nwp_loop(cfg, store), name="herbie_nwp"),
        asyncio.create_task(nws_gridpoint_loop(cfg, store, nws), name="nws_gridpoint"),
        asyncio.create_task(scoring_loop(cfg, store), name="scoring"),
        asyncio.create_task(bias_adjust_loop(cfg, store), name="bias_adjust"),
        asyncio.create_task(inference_loop(cfg, store, live), name="inference"),
        asyncio.create_task(_ds3m_shadow_wrapper(cfg, store, live), name="ds3m_shadow"),
    ]

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await asyncio.gather(
            wethr.close(), openmeteo.close(), nws.close(), kalshi_rest.close(),
        )
        store.close()
        log.info("Shutdown complete.")


def cli_entry() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    asyncio.run(main(config_path))


if __name__ == "__main__":
    cli_entry()
