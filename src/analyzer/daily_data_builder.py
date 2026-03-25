"""Daily data builder — assembles all collected data for a completed climate day.

Pulls every data source from the SQLite database for a target LST date and
packages it into a structured DailyDataPackage for LLM prompt construction.

Climate day bounds: midnight-midnight LST = 05:00Z to 05:00Z next day for KMIA
(UTC-5 fixed), consistent with climate_clock.py.

Usage:
    python -m analyzer.daily_data_builder --db miami_collector.db --date 2026-03-17
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

LST = timezone(timedelta(hours=-5), name="LST")
UTC = timezone.utc


def _climate_day_bounds_utc(target_date: str | date) -> tuple[datetime, datetime]:
    """Return (start_utc, end_utc) for a climate day in KMIA LST."""
    if isinstance(target_date, str):
        day = date.fromisoformat(target_date)
    else:
        day = target_date
    start_lst = datetime.combine(day, time.min, tzinfo=LST)
    end_lst = start_lst + timedelta(days=1)
    return start_lst.astimezone(UTC), end_lst.astimezone(UTC)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_to_lst(utc_str: str, offset: int = -5) -> tuple[str, int]:
    """Convert a UTC timestamp string to (LST time string, LST hour).

    Returns e.g. ("14:35", 14).
    """
    tz = timezone(timedelta(hours=offset))
    raw = utc_str.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    lst_dt = dt.astimezone(tz)
    return lst_dt.strftime("%H:%M"), lst_dt.hour


# ---------------------------------------------------------------------------
# Data package
# ---------------------------------------------------------------------------


@dataclass
class DailyDataPackage:
    """All collected data for a single climate day."""

    target_date: str
    station: str
    utc_start: str
    utc_end: str

    # Surface observations (adaptive-resolution)
    observations: list[dict] = field(default_factory=list)

    # Model forecasts (deduplicated to latest fetch per model/source)
    model_forecasts: list[dict] = field(default_factory=list)

    # Event settlements (CLI truth)
    event_settlements: list[dict] = field(default_factory=list)

    # Forward curves (all snapshots)
    forward_curves: list[dict] = field(default_factory=list)

    # Atmospheric data (CAPE, PW, radiation, soil, precip)
    atmospheric_data: list[dict] = field(default_factory=list)

    # Pressure levels (925/850/700/500 hPa)
    pressure_levels: list[dict] = field(default_factory=list)

    # FAWN observations (Homestead ag station)
    fawn_observations: list[dict] = field(default_factory=list)

    # Nearby ASOS station observations
    nearby_observations: list[dict] = field(default_factory=list)

    # SST buoy observations
    sst_observations: list[dict] = field(default_factory=list)

    # Kalshi market snapshots
    market_snapshots: list[dict] = field(default_factory=list)

    # Model consensus (bias-adjusted + weighted)
    model_consensus: list[dict] = field(default_factory=list)

    # Recent regime labels (previous 3 days)
    regime_labels: list[dict] = field(default_factory=list)

    def to_prompt_text(self, *, max_rows_per_section: int = 96) -> str:
        """Render the package as compact markdown for LLM consumption.

        Large tables (atmospheric, pressure levels, nearby, market snapshots)
        are sampled to keep the prompt within LLM context limits. Surface obs
        and FAWN obs are kept at full resolution since they're already adaptive.
        """
        parts: list[str] = []
        parts.append(f"# Climate Day Data: {self.station} {self.target_date}")
        parts.append(f"UTC window: {self.utc_start} to {self.utc_end}\n")

        # -- Event Settlements (truth) --
        parts.append("## Event Settlements (CLI Truth)")
        if self.event_settlements:
            parts.append("| market_type | actual_f |")
            parts.append("|---|---|")
            for r in self.event_settlements:
                parts.append(f"| {r['market_type']} | {_fmt(r['actual_value_f'])} |")
        else:
            parts.append("_No settlements recorded._")
        parts.append("")

        # -- Regime Labels (previous 3 days) --
        parts.append("## Recent Regime Labels (prior 3 days)")
        if self.regime_labels:
            parts.append("| date | regimes | path_class | confidence | phase |")
            parts.append("|---|---|---|---|---|")
            for r in self.regime_labels:
                parts.append(
                    f"| {r['target_date']} | {r['regimes_active']} "
                    f"| {r.get('path_class', '')} "
                    f"| {r.get('confidence_tags', '')} "
                    f"| {r.get('phase_summary', '')} |"
                )
        else:
            parts.append("_No recent regime labels._")
        parts.append("")

        # -- Model Consensus --
        parts.append("## Model Consensus (Bias-Adjusted)")
        if self.model_consensus:
            parts.append(
                "| market | model | source | raw_f | bias | adj_f | mae | weight | consensus_f | std_f | n |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in self.model_consensus:
                parts.append(
                    f"| {r.get('market_type', '')} "
                    f"| {r['model']} | {r['source']} "
                    f"| {_fmt(r.get('raw_forecast_f'))} "
                    f"| {_fmt(r.get('bias'))} "
                    f"| {_fmt(r.get('forecast_f'))} "
                    f"| {_fmt(r.get('mae'))} "
                    f"| {_fmt(r.get('final_weight'), 4)} "
                    f"| {_fmt(r.get('consensus_forecast_f'))} "
                    f"| {_fmt(r.get('consensus_std_f'))} "
                    f"| {r.get('n_models', '')} |"
                )
        else:
            parts.append("_No consensus data._")
        parts.append("")

        # -- Model Forecasts --
        parts.append("## Model Forecasts (Latest per Model/Source)")
        if self.model_forecasts:
            parts.append("| model | source | high_f | low_f | run_time | fetch_time |")
            parts.append("|---|---|---|---|---|---|")
            for r in self.model_forecasts:
                parts.append(
                    f"| {r['model']} | {r['source']} "
                    f"| {_fmt(r.get('forecast_high_f'))} "
                    f"| {_fmt(r.get('forecast_low_f'))} "
                    f"| {r.get('run_time', '')} "
                    f"| {r.get('fetch_time_utc', '')} |"
                )
        else:
            parts.append("_No model forecasts._")
        parts.append("")

        # -- Surface Observations (OHLC hourly with transition injection) --
        parts.append(f"## Surface Observations ({len(self.observations)} raw obs, OHLC hourly)")
        parts.append("_RAPID_CHANGE rows show exact sub-hour transition windows._")
        if self.observations:
            ohlc = _ohlc_aggregate_obs(self.observations)
            parts.append(
                "| LST | Temp_O | Temp_H | Temp_L | Temp_C | Dew_O | Dew_C "
                "| Wind_O→C | Spd | Gust | Pres_O→C | Sky% | Flag |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in ohlc:
                wo = _fmt(r.get("wind_open"), 0)
                wc = _fmt(r.get("wind_close"), 0)
                wind_str = f"{wo}→{wc}" if wo and wc and wo != wc else wo or wc or ""
                po = _fmt(r.get("pres_open"))
                pc = _fmt(r.get("pres_close"))
                pres_str = f"{po}→{pc}" if po and pc and po != pc else po or pc or ""
                parts.append(
                    f"| {r['label']} "
                    f"| {_fmt(r.get('temp_open'))} "
                    f"| {_fmt(r.get('temp_high'))} "
                    f"| {_fmt(r.get('temp_low'))} "
                    f"| {_fmt(r.get('temp_close'))} "
                    f"| {_fmt(r.get('dew_open'))} "
                    f"| {_fmt(r.get('dew_close'))} "
                    f"| {wind_str} "
                    f"| {_fmt(r.get('wind_speed_mean'))} "
                    f"| {_fmt(r.get('gust_max'))} "
                    f"| {pres_str} "
                    f"| {_fmt(r.get('sky_mode'), 0)} "
                    f"| {r.get('flag', '')} |"
                )
        else:
            parts.append("_No surface observations._")
        parts.append("")

        # -- Forward Curves (sampled) --
        parts.append(f"## Forward Curves ({len(self.forward_curves)} total, sampled)")
        if self.forward_curves:
            sampled_fc = _sample_rows(self.forward_curves, max_rows_per_section, "snapshot_time_utc")
            parts.append(
                "| snapshot | valid_hr | h_ahead | nbm | gfs | ecmwf | hrrr | nam "
                "| spread | cape | pw | precip% | solar | t850 | t925 |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sampled_fc:
                parts.append(
                    f"| {_short_ts(r.get('snapshot_time_utc', ''))} "
                    f"| {_short_ts(r.get('valid_hour_utc', ''))} "
                    f"| {r.get('hours_ahead', '')} "
                    f"| {_fmt(r.get('nbm_temp_f'))} "
                    f"| {_fmt(r.get('gfs_temp_f'))} "
                    f"| {_fmt(r.get('ecmwf_temp_f'))} "
                    f"| {_fmt(r.get('hrrr_temp_f'))} "
                    f"| {_fmt(r.get('nam_temp_f'))} "
                    f"| {_fmt(r.get('model_spread_f'))} "
                    f"| {_fmt(r.get('cape'), 0)} "
                    f"| {_fmt(r.get('pw_mm'))} "
                    f"| {_fmt(r.get('precip_prob'), 0)} "
                    f"| {_fmt(r.get('solar_wm2'), 0)} "
                    f"| {_fmt(r.get('temp_850_c'))} "
                    f"| {_fmt(r.get('temp_925_c'))} |"
                )
        else:
            parts.append("_No forward curves._")
        parts.append("")

        # -- Atmospheric Data (sampled) --
        parts.append(f"## Atmospheric Data ({len(self.atmospheric_data)} total, sampled)")
        if self.atmospheric_data:
            sampled_atmos = _sample_rows(self.atmospheric_data, max_rows_per_section, "valid_time_utc")
            parts.append(
                "| valid_utc | lst | cape | li | pw_mm | sw | dir_rad | diff_rad "
                "| blh | soil_t | soil_m | precip | rain | showers | precip% |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sampled_atmos:
                lst_str, _ = _utc_to_lst(r["valid_time_utc"])
                parts.append(
                    f"| {_short_ts(r['valid_time_utc'])} | {lst_str} "
                    f"| {_fmt(r.get('cape'), 0)} "
                    f"| {_fmt(r.get('lifted_index'))} "
                    f"| {_fmt(r.get('precipitable_water_mm'))} "
                    f"| {_fmt(r.get('shortwave_radiation'), 0)} "
                    f"| {_fmt(r.get('direct_radiation'), 0)} "
                    f"| {_fmt(r.get('diffuse_radiation'), 0)} "
                    f"| {_fmt(r.get('boundary_layer_height'), 0)} "
                    f"| {_fmt(r.get('soil_temperature_0_7cm'))} "
                    f"| {_fmt(r.get('soil_moisture_0_1cm'))} "
                    f"| {_fmt(r.get('precipitation_mm'))} "
                    f"| {_fmt(r.get('rain_mm'))} "
                    f"| {_fmt(r.get('showers_mm'))} "
                    f"| {_fmt(r.get('precipitation_probability'), 0)} |"
                )
        else:
            parts.append("_No atmospheric data._")
        parts.append("")

        # -- Pressure Levels (sampled) --
        parts.append(f"## Pressure Levels ({len(self.pressure_levels)} total, sampled)")
        if self.pressure_levels:
            sampled_plev = _sample_rows(self.pressure_levels, max_rows_per_section, "valid_time_utc")
            parts.append(
                "| valid_utc | lst | t925 | w925 | d925 | gp925 "
                "| t850 | w850 | d850 | rh850 | t700 | w700 | d700 | rh700 "
                "| t500 | w500 | d500 |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sampled_plev:
                lst_str, _ = _utc_to_lst(r["valid_time_utc"])
                parts.append(
                    f"| {_short_ts(r['valid_time_utc'])} | {lst_str} "
                    f"| {_fmt(r.get('temp_925_c'))} "
                    f"| {_fmt(r.get('wind_speed_925'))} "
                    f"| {_fmt(r.get('wind_dir_925'), 0)} "
                    f"| {_fmt(r.get('geopotential_925'), 0)} "
                    f"| {_fmt(r.get('temp_850_c'))} "
                    f"| {_fmt(r.get('wind_speed_850'))} "
                    f"| {_fmt(r.get('wind_dir_850'), 0)} "
                    f"| {_fmt(r.get('rh_850'), 0)} "
                    f"| {_fmt(r.get('temp_700_c'))} "
                    f"| {_fmt(r.get('wind_speed_700'))} "
                    f"| {_fmt(r.get('wind_dir_700'), 0)} "
                    f"| {_fmt(r.get('rh_700'), 0)} "
                    f"| {_fmt(r.get('temp_500_c'))} "
                    f"| {_fmt(r.get('wind_speed_500'))} "
                    f"| {_fmt(r.get('wind_dir_500'), 0)} |"
                )
        else:
            parts.append("_No pressure level data._")
        parts.append("")

        # -- FAWN Observations --
        parts.append("## FAWN Observations (Homestead)")
        if self.fawn_observations:
            parts.append(
                "| time_utc | lst | temp_f | dew_f | wind | gust | dir "
                "| solar_wm2 | rain_mm | soil_c | soil_f |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in self.fawn_observations:
                lst_str, _ = _utc_to_lst(r["timestamp_utc"])
                parts.append(
                    f"| {_short_ts(r['timestamp_utc'])} | {lst_str} "
                    f"| {_fmt(r.get('air_temp_f'))} "
                    f"| {_fmt(r.get('dew_point_f'))} "
                    f"| {_fmt(r.get('wind_speed_mph'))} "
                    f"| {_fmt(r.get('wind_gust_mph'))} "
                    f"| {_fmt(r.get('wind_direction_deg'), 0)} "
                    f"| {_fmt(r.get('solar_radiation_wm2'), 0)} "
                    f"| {_fmt(r.get('rain_mm'))} "
                    f"| {_fmt(r.get('soil_temp_c'))} "
                    f"| {_fmt(r.get('soil_temp_f'))} |"
                )
        else:
            parts.append("_No FAWN observations._")
        parts.append("")

        # -- Nearby Observations (OHLC per station per hour) --
        parts.append(f"## Nearby Station Observations ({len(self.nearby_observations)} raw, hourly per station)")
        if self.nearby_observations:
            nearby_agg = _ohlc_aggregate_nearby(self.nearby_observations)
            parts.append(
                "| stid | name | dist | hour | temp_hi | temp_lo | delta_mean | n |"
            )
            parts.append("|---|---|---|---|---|---|---|---|")
            for r in nearby_agg:
                parts.append(
                    f"| {r.get('stid', '')} "
                    f"| {r.get('name', '')} "
                    f"| {_fmt(r.get('dist_mi'))} "
                    f"| {r.get('hour_lst', ''):02d}:00 "
                    f"| {_fmt(r.get('temp_high'))} "
                    f"| {_fmt(r.get('temp_low'))} "
                    f"| {_fmt(r.get('delta_mean'))} "
                    f"| {r.get('n_obs', '')} |"
                )
        else:
            parts.append("_No nearby observations._")
        parts.append("")

        # -- SST Observations --
        parts.append("## SST Buoy Observations")
        if self.sst_observations:
            parts.append(
                "| time_utc | lst | station | name | dist_mi "
                "| water_c | water_f | air_c | wind_mps | wind_dir | pres |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in self.sst_observations:
                lst_str, _ = _utc_to_lst(r["timestamp_utc"])
                parts.append(
                    f"| {_short_ts(r['timestamp_utc'])} | {lst_str} "
                    f"| {r.get('station_id', '')} "
                    f"| {r.get('name', '')} "
                    f"| {_fmt(r.get('distance_mi'))} "
                    f"| {_fmt(r.get('water_temp_c'))} "
                    f"| {_fmt(r.get('water_temp_f'))} "
                    f"| {_fmt(r.get('air_temp_c'))} "
                    f"| {_fmt(r.get('wind_speed_mps'))} "
                    f"| {_fmt(r.get('wind_dir_deg'), 0)} "
                    f"| {_fmt(r.get('pressure_hpa'))} |"
                )
        else:
            parts.append("_No SST observations._")
        parts.append("")

        # -- Market Snapshots (sampled per ticker — price evolution) --
        parts.append(f"## Market Snapshots ({len(self.market_snapshots)} raw, sampled)")
        if self.market_snapshots:
            sampled_mkt = _ohlc_aggregate_market(self.market_snapshots)
            parts.append(
                "| time | ticker | type | floor | cap "
                "| bid_c | ask_c | last_c | vol | spread |"
            )
            parts.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in sampled_mkt:
                parts.append(
                    f"| {_short_ts(r.get('snapshot_time', ''))} "
                    f"| {r.get('ticker', '')} "
                    f"| {r.get('market_type', '')} "
                    f"| {_fmt(r.get('floor_strike'))} "
                    f"| {_fmt(r.get('cap_strike'))} "
                    f"| {_fmt(r.get('best_yes_bid_cents'))} "
                    f"| {_fmt(r.get('best_yes_ask_cents'))} "
                    f"| {_fmt(r.get('last_price_cents'))} "
                    f"| {r.get('volume', '')} "
                    f"| {_fmt(r.get('spread_cents'))} |"
                )
        else:
            parts.append("_No market snapshots._")
        parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _sample_rows(rows: list[dict], max_rows: int, time_key: str = "timestamp_utc") -> list[dict]:
    """Evenly sample rows by time to reduce volume while preserving temporal coverage."""
    if len(rows) <= max_rows:
        return rows
    step = len(rows) / max_rows
    indices = [int(i * step) for i in range(max_rows)]
    if indices[-1] != len(rows) - 1:
        indices[-1] = len(rows) - 1
    return [rows[i] for i in indices]


def _sample_market_snapshots(rows: list[dict], max_per_ticker: int = 12) -> list[dict]:
    """Sample market snapshots: keep key timestamps per ticker to show price evolution."""
    from collections import defaultdict
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ticker[r.get("ticker", "")].append(r)
    result = []
    for ticker in sorted(by_ticker):
        ticker_rows = by_ticker[ticker]
        result.extend(_sample_rows(ticker_rows, max_per_ticker, "snapshot_time"))
    return result


# ---------------------------------------------------------------------------
# OHLC aggregation with transition injection
# ---------------------------------------------------------------------------

def _ohlc_aggregate_obs(
    rows: list[dict],
    utc_offset: int = -5,
) -> list[dict]:
    """Aggregate surface obs into OHLC-style hourly rows with transition injection.

    For each LST hour, produces one row with open/high/low/close for temp, dew,
    pressure, plus wind direction range and max gust. When a rapid change is
    detected within an hour, an extra RAPID_CHANGE row is injected showing
    the exact transition window (start→end values over the sub-hour period).
    """
    if not rows:
        return []

    from collections import defaultdict

    # Group by LST hour
    by_hour: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        _, hr = _utc_to_lst(r.get("timestamp_utc", ""), utc_offset)
        by_hour[hr].append(r)

    result = []
    for hr in sorted(by_hour):
        obs_list = by_hour[hr]
        if not obs_list:
            continue

        # Build OHLC row
        temps = [o.get("temperature_f") for o in obs_list if o.get("temperature_f") is not None]
        dews = [o.get("dew_point_f") for o in obs_list if o.get("dew_point_f") is not None]
        pressures = [o.get("pressure_hpa") for o in obs_list if o.get("pressure_hpa") is not None]
        winds = [o.get("wind_heading_deg") for o in obs_list if o.get("wind_heading_deg") is not None]
        gusts = [o.get("wind_gust_mph") for o in obs_list if o.get("wind_gust_mph") is not None]
        speeds = [o.get("wind_speed_mph") for o in obs_list if o.get("wind_speed_mph") is not None]

        ohlc = {
            "hour_lst": hr,
            "label": f"{hr:02d}:00",
            "temp_open": temps[0] if temps else None,
            "temp_high": max(temps) if temps else None,
            "temp_low": min(temps) if temps else None,
            "temp_close": temps[-1] if temps else None,
            "dew_open": dews[0] if dews else None,
            "dew_close": dews[-1] if dews else None,
            "pres_open": pressures[0] if pressures else None,
            "pres_close": pressures[-1] if pressures else None,
            "wind_open": winds[0] if winds else None,
            "wind_close": winds[-1] if winds else None,
            "wind_speed_mean": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "gust_max": max(gusts) if gusts else None,
            "sky_mode": _mode([o.get("sky_cover_pct") for o in obs_list if o.get("sky_cover_pct") is not None]),
            "flag": "",
        }
        result.append(ohlc)

        # Check for rapid transitions within this hour
        transitions = _detect_transitions(obs_list, utc_offset)
        for t in transitions:
            result.append(t)

    # Sort by hour then by transition order
    result.sort(key=lambda x: (x["hour_lst"], 0 if x["flag"] == "" else 1))
    return result


def _detect_transitions(obs_list: list[dict], utc_offset: int = -5) -> list[dict]:
    """Find rapid change windows within an hour's obs and return transition rows."""
    transitions = []
    if len(obs_list) < 3:
        return transitions

    # Sliding window: compare each obs to the one 3-5 positions ahead
    for i in range(len(obs_list) - 2):
        # Look ahead 2-5 obs (roughly 10-25 minutes)
        for j in range(i + 2, min(i + 6, len(obs_list))):
            prev, curr = obs_list[i], obs_list[j]
            pt = prev.get("temperature_f")
            ct = curr.get("temperature_f")
            pd = prev.get("dew_point_f")
            cd = curr.get("dew_point_f")
            pw = prev.get("wind_heading_deg")
            cw = curr.get("wind_heading_deg")
            pp = prev.get("pressure_hpa")
            cp = curr.get("pressure_hpa")

            temp_change = abs(ct - pt) if pt is not None and ct is not None else 0
            dew_change = abs(cd - pd) if pd is not None and cd is not None else 0
            wind_change = _angular_diff(pw, cw)
            pres_change = abs(cp - pp) if pp is not None and cp is not None else 0

            # Threshold: flag any meaningful sub-hour change.
            # 3°F temp, 3°F dew, 45° wind, or 0.5 hPa pressure.
            # Let the AI decide what's significant — better noisy than missing a precursor.
            if temp_change >= 3.0 or dew_change >= 3.0 or wind_change >= 45 or pres_change >= 0.5:
                start_ts = prev.get("timestamp_utc", "")
                end_ts = curr.get("timestamp_utc", "")
                start_lst, hr = _utc_to_lst(start_ts, utc_offset)
                end_lst, _ = _utc_to_lst(end_ts, utc_offset)

                gusts_window = [o.get("wind_gust_mph") for o in obs_list[i:j+1]
                                if o.get("wind_gust_mph") is not None]

                transition = {
                    "hour_lst": hr,
                    "label": f"{start_lst}→{end_lst}",
                    "temp_open": pt,
                    "temp_high": max(o.get("temperature_f", -999) for o in obs_list[i:j+1]
                                     if o.get("temperature_f") is not None) if pt else None,
                    "temp_low": min(o.get("temperature_f", 999) for o in obs_list[i:j+1]
                                    if o.get("temperature_f") is not None) if ct else None,
                    "temp_close": ct,
                    "dew_open": pd,
                    "dew_close": cd,
                    "pres_open": pp,
                    "pres_close": cp,
                    "wind_open": pw,
                    "wind_close": cw,
                    "wind_speed_mean": None,
                    "gust_max": max(gusts_window) if gusts_window else None,
                    "sky_mode": None,
                    "flag": "RAPID_CHANGE",
                }
                transitions.append(transition)
                return transitions  # One transition per hour is enough

    return transitions


def _mode(values: list) -> Any:
    """Return the most common value, or None."""
    if not values:
        return None
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


def _ohlc_aggregate_nearby(
    rows: list[dict],
    utc_offset: int = -5,
) -> list[dict]:
    """Aggregate nearby station obs into per-station OHLC hourly summaries.

    Returns one row per station per hour with temp OHLC, delta range, and wind summary.
    """
    if not rows:
        return []

    from collections import defaultdict

    # Group by station then hour
    by_station: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        stid = r.get("stid", "")
        _, hr = _utc_to_lst(r.get("timestamp_utc", ""), utc_offset)
        by_station[stid][hr].append(r)

    result = []
    for stid in sorted(by_station):
        station_hours = by_station[stid]
        first_obs = next(iter(next(iter(station_hours.values()))))
        name = first_obs.get("name", "")
        dist = first_obs.get("distance_mi")

        for hr in sorted(station_hours):
            obs_list = station_hours[hr]
            temps = [o.get("air_temp_f") for o in obs_list if o.get("air_temp_f") is not None]
            deltas = [o.get("temp_delta_vs_kmia") for o in obs_list if o.get("temp_delta_vs_kmia") is not None]

            result.append({
                "stid": stid,
                "name": name,
                "dist_mi": dist,
                "hour_lst": hr,
                "temp_high": max(temps) if temps else None,
                "temp_low": min(temps) if temps else None,
                "delta_mean": round(sum(deltas) / len(deltas), 1) if deltas else None,
                "n_obs": len(obs_list),
            })

    return result


def _ohlc_aggregate_market(
    rows: list[dict],
    utc_offset: int = -5,
    snapshots_per_ticker: int = 8,
) -> list[dict]:
    """Aggregate market snapshots to key price points per ticker.

    For each ticker, keeps: first, last, highest bid, lowest bid, and
    evenly spaced samples showing price evolution.
    """
    if not rows:
        return []

    from collections import defaultdict

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ticker[r.get("ticker", "")].append(r)

    result = []
    for ticker in sorted(by_ticker):
        ticker_rows = by_ticker[ticker]
        if not ticker_rows:
            continue

        # Always include first and last
        sampled = _sample_rows(ticker_rows, snapshots_per_ticker, "snapshot_time")

        for r in sampled:
            result.append(r)

    return result


def _fmt(value: Any, decimals: int = 1) -> str:
    """Format a numeric value compactly, or return empty string for None."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        v = float(value)
        if decimals == 0:
            return str(int(round(v)))
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def _short_ts(ts: str) -> str:
    """Shorten an ISO timestamp to HH:MM for compactness."""
    if not ts:
        return ""
    # Handle both "2026-03-17T05:00:00Z" and "2026-03-17 05:00:00" forms
    try:
        if "T" in ts:
            return ts.split("T")[1][:5]
        parts = ts.split(" ")
        if len(parts) >= 2:
            return parts[1][:5]
    except (IndexError, AttributeError):
        pass
    return ts[:16]


# ---------------------------------------------------------------------------
# Densification thresholds
# ---------------------------------------------------------------------------

_TEMP_CHANGE_THRESHOLD = 1.0       # degrees F
_DEW_CHANGE_THRESHOLD = 1.5        # degrees F
_WIND_SHIFT_THRESHOLD = 30.0       # degrees
_PRESSURE_CHANGE_THRESHOLD = 0.5   # hPa


def _angular_diff(a: float | None, b: float | None) -> float:
    """Smallest angular difference between two headings."""
    if a is None or b is None:
        return 0.0
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _significant_change(prev: dict, curr: dict) -> bool:
    """Detect whether a tangible change occurred between two obs samples."""
    pt = prev.get("temperature_f")
    ct = curr.get("temperature_f")
    if pt is not None and ct is not None and abs(ct - pt) > _TEMP_CHANGE_THRESHOLD:
        return True

    pd = prev.get("dew_point_f")
    cd = curr.get("dew_point_f")
    if pd is not None and cd is not None and abs(cd - pd) > _DEW_CHANGE_THRESHOLD:
        return True

    pw = prev.get("wind_heading_deg")
    cw = curr.get("wind_heading_deg")
    if _angular_diff(pw, cw) > _WIND_SHIFT_THRESHOLD:
        return True

    pp = prev.get("pressure_hpa")
    cp = curr.get("pressure_hpa")
    if pp is not None and cp is not None and abs(cp - pp) > _PRESSURE_CHANGE_THRESHOLD:
        return True

    return False


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_OBS_COLUMNS = [
    "timestamp_utc", "temperature_f", "dew_point_f", "wind_speed_mph",
    "wind_heading_deg", "wind_gust_mph", "pressure_hpa", "sky_cover_pct",
    "sky_cover_code", "visibility_miles", "source",
]


class DailyDataBuilder:
    """Assembles all collected data for a completed climate day."""

    def __init__(
        self,
        db_path: str,
        station: str = "KMIA",
        utc_offset: int = -5,
    ):
        self.db_path = db_path
        self.station = station
        self.utc_offset = utc_offset

    # -- public API --

    def build(self, target_date: str) -> DailyDataPackage:
        """Connect to DB, run all queries, return the assembled package."""
        start_utc, end_utc = _climate_day_bounds_utc(target_date)
        start_iso = _iso_z(start_utc)
        end_iso = _iso_z(end_utc)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            pkg = DailyDataPackage(
                target_date=target_date,
                station=self.station,
                utc_start=start_iso,
                utc_end=end_iso,
            )

            pkg.observations = self._query_observations(conn, target_date, start_iso, end_iso)
            pkg.model_forecasts = self._query_model_forecasts(conn, target_date)
            pkg.event_settlements = self._query_event_settlements(conn, target_date)
            pkg.forward_curves = self._query_forward_curves(conn, target_date)
            pkg.atmospheric_data = self._query_atmospheric_data(conn, start_iso, end_iso)
            pkg.pressure_levels = self._query_pressure_levels(conn, start_iso, end_iso)
            pkg.fawn_observations = self._query_fawn(conn, start_iso, end_iso)
            pkg.nearby_observations = self._query_nearby(conn, start_iso, end_iso)
            pkg.sst_observations = self._query_sst(conn, start_iso, end_iso)
            pkg.market_snapshots = self._query_market_snapshots(conn, target_date)
            pkg.model_consensus = self._query_model_consensus(conn, target_date)
            pkg.regime_labels = self._query_recent_regime_labels(conn, target_date)

            return pkg
        finally:
            conn.close()

    # -- queries --

    def _query_observations(
        self,
        conn: sqlite3.Connection,
        target_date: str,
        start_iso: str,
        end_iso: str,
    ) -> list[dict]:
        """Query surface obs with adaptive densification.

        1. Pull all obs for the climate day.
        2. Sample at 15-min intervals.
        3. Scan for significant changes between consecutive samples.
        4. Where change detected, pull full-resolution obs for that window.
        """
        cols = ", ".join(_OBS_COLUMNS)
        all_rows = conn.execute(
            f"""SELECT {cols} FROM observations
                WHERE station = ? AND lst_date = ?
                ORDER BY timestamp_utc""",
            (self.station, target_date),
        ).fetchall()

        if not all_rows:
            return []

        all_obs = [dict(r) for r in all_rows]

        # Build a lookup by timestamp for full-resolution retrieval
        obs_by_ts: dict[str, dict] = {}
        for o in all_obs:
            obs_by_ts[o["timestamp_utc"]] = o

        # Sample at 15-min intervals: pick the obs nearest to each 15-min mark
        sampled = self._sample_15min(all_obs, start_iso, end_iso)

        if len(sampled) < 2:
            return sampled

        # Detect transitions and densify
        densified_windows: list[tuple[str, str]] = []
        for i in range(1, len(sampled)):
            if _significant_change(sampled[i - 1], sampled[i]):
                # Window from previous sample to current sample
                densified_windows.append(
                    (sampled[i - 1]["timestamp_utc"], sampled[i]["timestamp_utc"])
                )

        if not densified_windows:
            return sampled

        # Merge overlapping windows
        merged = self._merge_windows(densified_windows)

        # Collect full-resolution obs for each densified window
        dense_ts: set[str] = set()
        for win_start, win_end in merged:
            for o in all_obs:
                ts = o["timestamp_utc"]
                if win_start <= ts <= win_end:
                    dense_ts.add(ts)

        # Combine: sampled timestamps + densified timestamps
        all_ts = {o["timestamp_utc"] for o in sampled} | dense_ts
        result = [obs_by_ts[ts] for ts in sorted(all_ts) if ts in obs_by_ts]
        return result

    def _sample_15min(
        self,
        obs: list[dict],
        start_iso: str,
        end_iso: str,
    ) -> list[dict]:
        """Pick one obs per 15-minute slot, nearest to the slot center."""
        if not obs:
            return []

        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

        # Generate 15-min marks
        marks: list[datetime] = []
        t = start_dt
        while t < end_dt:
            marks.append(t)
            t += timedelta(minutes=15)

        # Parse all obs timestamps once
        parsed: list[tuple[datetime, dict]] = []
        for o in obs:
            raw = o["timestamp_utc"].strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            parsed.append((dt, o))

        sampled: list[dict] = []
        used: set[int] = set()

        for mark in marks:
            best_idx = -1
            best_diff = timedelta(minutes=8)  # max 7.5 min from slot center
            for idx, (dt, _) in enumerate(parsed):
                if idx in used:
                    continue
                diff = abs(dt - mark)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            if best_idx >= 0:
                used.add(best_idx)
                sampled.append(parsed[best_idx][1])

        return sampled

    @staticmethod
    def _merge_windows(windows: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge overlapping or adjacent time windows."""
        if not windows:
            return []
        sorted_w = sorted(windows)
        merged = [sorted_w[0]]
        for start, end in sorted_w[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _query_model_forecasts(self, conn: sqlite3.Connection, target_date: str) -> list[dict]:
        """Deduplicated to latest fetch per model/source."""
        rows = conn.execute(
            """SELECT model, source, forecast_high_f, forecast_low_f,
                      run_time, fetch_time_utc, source_record_json
               FROM model_forecasts
               WHERE station = ? AND forecast_date = ?
                 AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
               ORDER BY model, source, id DESC""",
            (self.station, target_date),
        ).fetchall()

        # Dedupe: keep only the latest row per (model, source)
        seen: set[tuple[str, str]] = set()
        result: list[dict] = []
        for r in rows:
            key = (r["model"], r["source"])
            if key not in seen:
                seen.add(key)
                result.append(dict(r))
        return result

    def _query_event_settlements(self, conn: sqlite3.Connection, target_date: str) -> list[dict]:
        rows = conn.execute(
            """SELECT market_type, actual_value_f, settlement_source, raw_text
               FROM event_settlements
               WHERE station = ? AND settlement_date = ?""",
            (self.station, target_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_forward_curves(self, conn: sqlite3.Connection, target_date: str) -> list[dict]:
        rows = conn.execute(
            """SELECT snapshot_time_utc, valid_hour_utc, hours_ahead,
                      nbm_temp_f, gfs_temp_f, ecmwf_temp_f, hrrr_temp_f, nam_temp_f,
                      model_min_f, model_max_f, model_spread_f,
                      cape, pw_mm, precip_prob, precip_mm, solar_wm2,
                      temp_850_c, temp_925_c, wind_dir_850, wind_speed_850
               FROM forward_curves
               WHERE station = ? AND target_date = ?
               ORDER BY snapshot_time_utc, valid_hour_utc""",
            (self.station, target_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_atmospheric_data(
        self, conn: sqlite3.Connection, start_iso: str, end_iso: str,
    ) -> list[dict]:
        rows = conn.execute(
            """SELECT valid_time_utc, cape, lifted_index,
                      precipitable_water_mm, shortwave_radiation,
                      direct_radiation, diffuse_radiation,
                      boundary_layer_height, soil_temperature_0_7cm,
                      soil_moisture_0_1cm, precipitation_mm, rain_mm,
                      showers_mm, precipitation_probability
               FROM atmospheric_data
               WHERE station = ?
                 AND valid_time_utc >= ? AND valid_time_utc < ?
               ORDER BY valid_time_utc""",
            (self.station, start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_pressure_levels(
        self, conn: sqlite3.Connection, start_iso: str, end_iso: str,
    ) -> list[dict]:
        rows = conn.execute(
            """SELECT valid_time_utc,
                      temp_925_c, wind_speed_925, wind_dir_925, geopotential_925,
                      temp_850_c, wind_speed_850, wind_dir_850, geopotential_850,
                      temp_700_c, wind_speed_700, wind_dir_700, geopotential_700,
                      rh_850, rh_700,
                      temp_500_c, wind_speed_500, wind_dir_500
               FROM pressure_levels
               WHERE station = ?
                 AND valid_time_utc >= ? AND valid_time_utc < ?
               ORDER BY valid_time_utc""",
            (self.station, start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_fawn(
        self, conn: sqlite3.Connection, start_iso: str, end_iso: str,
    ) -> list[dict]:
        rows = conn.execute(
            """SELECT timestamp_utc, air_temp_f, dew_point_f,
                      wind_speed_mph, wind_gust_mph, wind_direction_deg,
                      solar_radiation_wm2, rain_mm, soil_temp_c, soil_temp_f
               FROM fawn_observations
               WHERE timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY timestamp_utc""",
            (start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_nearby(
        self, conn: sqlite3.Connection, start_iso: str, end_iso: str,
    ) -> list[dict]:
        """Join nearby_observations with nearby_stations for distance/bearing."""
        rows = conn.execute(
            """SELECT o.timestamp_utc, o.stid, o.name,
                      COALESCE(s.distance_mi, o.distance_mi) AS distance_mi,
                      s.bearing_deg,
                      o.air_temp_f, o.temp_delta_vs_kmia,
                      o.wind_speed_mph, o.wind_direction_deg, o.wind_gust_mph,
                      o.pressure_slp_hpa, o.dew_point_f, o.sky_cover_code
               FROM nearby_observations o
               LEFT JOIN nearby_stations s ON o.stid = s.stid
               WHERE o.timestamp_utc >= ? AND o.timestamp_utc < ?
               ORDER BY o.timestamp_utc, COALESCE(s.distance_mi, o.distance_mi)""",
            (start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_sst(
        self, conn: sqlite3.Connection, start_iso: str, end_iso: str,
    ) -> list[dict]:
        rows = conn.execute(
            """SELECT station_id, name, timestamp_utc,
                      water_temp_c, water_temp_f, air_temp_c,
                      wind_speed_mps, wind_dir_deg, pressure_hpa, distance_mi
               FROM sst_observations
               WHERE timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY timestamp_utc, station_id""",
            (start_iso, end_iso),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_market_snapshots(self, conn: sqlite3.Connection, target_date: str) -> list[dict]:
        rows = conn.execute(
            """SELECT ticker, event_ticker, market_type, forecast_date,
                      floor_strike, cap_strike, best_yes_bid_cents,
                      best_yes_ask_cents, last_price_cents, volume,
                      spread_cents, snapshot_time
               FROM market_snapshots
               WHERE forecast_date = ?
               ORDER BY snapshot_time, ticker""",
            (target_date,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_model_consensus(self, conn: sqlite3.Connection, target_date: str) -> list[dict]:
        rows = conn.execute(
            """SELECT market_type, model, source,
                      raw_forecast_f, bias, forecast_f, mae,
                      skill_weight, final_weight,
                      consensus_forecast_f, consensus_std_f, n_models
               FROM model_consensus
               WHERE station = ? AND forecast_date = ?
               ORDER BY market_type, final_weight DESC""",
            (self.station, target_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_recent_regime_labels(
        self, conn: sqlite3.Connection, target_date: str,
    ) -> list[dict]:
        """Get regime labels for the 3 days preceding target_date."""
        day = date.fromisoformat(target_date)
        prior_dates = [(day - timedelta(days=i)).isoformat() for i in range(1, 4)]
        placeholders = ",".join("?" for _ in prior_dates)
        rows = conn.execute(
            f"""SELECT target_date, regimes_active, path_class,
                       confidence_tags, phase_summary, review_source
                FROM regime_labels
                WHERE station = ? AND target_date IN ({placeholders})
                ORDER BY target_date DESC""",
            (self.station, *prior_dates),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build daily data package and print prompt text.",
    )
    parser.add_argument("--db", required=True, help="Path to miami_collector.db")
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--station", default="KMIA", help="Station ID (default: KMIA)")
    args = parser.parse_args()

    builder = DailyDataBuilder(db_path=args.db, station=args.station)
    pkg = builder.build(args.date)

    # Print summary counts
    print(f"# Data counts for {args.station} {args.date}", file=sys.stderr)
    print(f"  observations:      {len(pkg.observations)}", file=sys.stderr)
    print(f"  model_forecasts:   {len(pkg.model_forecasts)}", file=sys.stderr)
    print(f"  event_settlements: {len(pkg.event_settlements)}", file=sys.stderr)
    print(f"  forward_curves:    {len(pkg.forward_curves)}", file=sys.stderr)
    print(f"  atmospheric_data:  {len(pkg.atmospheric_data)}", file=sys.stderr)
    print(f"  pressure_levels:   {len(pkg.pressure_levels)}", file=sys.stderr)
    print(f"  fawn_observations: {len(pkg.fawn_observations)}", file=sys.stderr)
    print(f"  nearby_obs:        {len(pkg.nearby_observations)}", file=sys.stderr)
    print(f"  sst_observations:  {len(pkg.sst_observations)}", file=sys.stderr)
    print(f"  market_snapshots:  {len(pkg.market_snapshots)}", file=sys.stderr)
    print(f"  model_consensus:   {len(pkg.model_consensus)}", file=sys.stderr)
    print(f"  regime_labels:     {len(pkg.regime_labels)}", file=sys.stderr)
    print(file=sys.stderr)

    # Print the prompt text to stdout
    print(pkg.to_prompt_text())


if __name__ == "__main__":
    main()
