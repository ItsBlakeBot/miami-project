"""Replay comparison utility: BOCPD-enabled detector vs CUSUM-only detector.

Phase mapping:
- T2.1 BOCPD prototype (comparison bundle)

Runs both detector configurations over historical observations and emits a
compact comparison report with fire counts, simple false-positive proxy, and
latency-to-event proxy metrics.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.changepoint_detector import ChangeDetector
from engine.replay_context import parse_utc

UTC = timezone.utc


@dataclass(frozen=True)
class FirePoint:
    ts: datetime
    layer: int


@dataclass(frozen=True)
class CompareStats:
    total_obs: int
    total_events: int
    total_fires: int
    fires_layer1: int
    fires_layer2: int
    fires_layer3: int
    matched_events: int
    missed_events: int
    median_latency_minutes: float | None
    false_positive_proxy: int


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _event_times(rows: list[sqlite3.Row], min_spacing_minutes: float = 30.0) -> list[datetime]:
    """Simple proxy event detector from abrupt observed shifts."""
    out: list[datetime] = []
    prev: sqlite3.Row | None = None

    for row in rows:
        ts = parse_utc(row["timestamp_utc"])
        if ts is None:
            continue

        trigger = False
        if prev is not None:
            def _delta(key: str) -> float | None:
                a = prev[key]
                b = row[key]
                if a is None or b is None:
                    return None
                return float(b) - float(a)

            d_temp = _delta("temperature_f")
            d_dew = _delta("dew_point_f")
            d_pres = _delta("pressure_hpa")
            d_dir = _delta("wind_heading_deg")

            if d_temp is not None and abs(d_temp) >= 2.5:
                trigger = True
            if d_dew is not None and abs(d_dew) >= 3.5:
                trigger = True
            if d_pres is not None and abs(d_pres) >= 0.7:
                trigger = True
            if d_dir is not None and abs(d_dir) >= 45.0:
                trigger = True

        if trigger:
            if not out or (ts - out[-1]).total_seconds() / 60.0 >= min_spacing_minutes:
                out.append(ts)

        prev = row

    return out


def _run_detector(
    rows: list[sqlite3.Row],
    *,
    use_bocpd: bool,
    db_path: str,
    station: str,
) -> tuple[list[FirePoint], int]:
    detector = ChangeDetector(use_bocpd=use_bocpd)
    detector.fit_diurnal(db_path, station=station)
    detector.reset()

    fires: list[FirePoint] = []
    total_obs = 0
    prev_ts: datetime | None = None

    for row in rows:
        ts = parse_utc(row["timestamp_utc"])
        if ts is None:
            continue
        total_obs += 1

        if prev_ts is None:
            delta_minutes = 5.0
        else:
            delta_minutes = max(0.5, min(60.0, (ts - prev_ts).total_seconds() / 60.0))
        prev_ts = ts

        hour_lst = (ts.hour - 5) % 24 + ts.minute / 60.0

        state = detector.update(
            {
                "temp_f": row["temperature_f"],
                "dew_f": row["dew_point_f"],
                "pressure_hpa": row["pressure_hpa"],
                "wind_speed_mph": row["wind_speed_mph"],
                "wind_gust_mph": row["wind_gust_mph"],
                "wind_dir_deg": row["wind_heading_deg"],
                "sky_cover_pct": row["sky_cover_pct"],
            },
            hour_lst=hour_lst,
            minutes_elapsed=delta_minutes,
        )

        if state.fired:
            fires.append(FirePoint(ts=ts, layer=int(state.layer)))

    return fires, total_obs


def _score(events: list[datetime], fires: list[FirePoint], window_minutes: float = 120.0) -> tuple[int, int, list[float], int]:
    matched = 0
    missed = 0
    latencies: list[float] = []

    fire_times = [f.ts for f in fires]

    for e in events:
        candidates = [f for f in fire_times if e <= f <= e + timedelta(minutes=window_minutes)]
        if not candidates:
            missed += 1
            continue
        matched += 1
        latency = (min(candidates) - e).total_seconds() / 60.0
        latencies.append(float(latency))

    false_positive = 0
    for f in fire_times:
        nearest = min((abs((f - e).total_seconds()) / 60.0 for e in events), default=None)
        if nearest is None or nearest > window_minutes:
            false_positive += 1

    return matched, missed, latencies, false_positive


def _collect_dates(conn: sqlite3.Connection, station: str, start_date: str | None, end_date: str | None) -> list[str]:
    sql = """SELECT DISTINCT settlement_date
             FROM event_settlements
             WHERE station = ?"""
    params: list[object] = [station]
    if start_date:
        sql += " AND settlement_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND settlement_date <= ?"
        params.append(end_date)
    sql += " ORDER BY settlement_date"

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [str(r[0]) for r in rows if r[0]]


def run_compare(
    db_path: str | Path,
    *,
    station: str = "KMIA",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        dates = _collect_dates(conn, station, start_date, end_date)

        all_rows: list[sqlite3.Row] = []
        for d in dates:
            rows = conn.execute(
                """SELECT timestamp_utc, temperature_f, dew_point_f, pressure_hpa,
                          wind_speed_mph, wind_gust_mph, wind_heading_deg, sky_cover_pct
                   FROM observations
                   WHERE station = ? AND lst_date = ?
                   ORDER BY timestamp_utc""",
                (station, d),
            ).fetchall()
            all_rows.extend(rows)
    finally:
        conn.close()

    events = _event_times(all_rows)

    fires_bocpd, total_obs_bocpd = _run_detector(
        all_rows,
        use_bocpd=True,
        db_path=str(db_path),
        station=station,
    )
    fires_cusum, total_obs_cusum = _run_detector(
        all_rows,
        use_bocpd=False,
        db_path=str(db_path),
        station=station,
    )

    m_b, miss_b, lat_b, fp_b = _score(events, fires_bocpd)
    m_c, miss_c, lat_c, fp_c = _score(events, fires_cusum)

    stats_b = CompareStats(
        total_obs=total_obs_bocpd,
        total_events=len(events),
        total_fires=len(fires_bocpd),
        fires_layer1=sum(1 for f in fires_bocpd if f.layer == 1),
        fires_layer2=sum(1 for f in fires_bocpd if f.layer == 2),
        fires_layer3=sum(1 for f in fires_bocpd if f.layer == 3),
        matched_events=m_b,
        missed_events=miss_b,
        median_latency_minutes=_median(lat_b),
        false_positive_proxy=fp_b,
    )
    stats_c = CompareStats(
        total_obs=total_obs_cusum,
        total_events=len(events),
        total_fires=len(fires_cusum),
        fires_layer1=sum(1 for f in fires_cusum if f.layer == 1),
        fires_layer2=sum(1 for f in fires_cusum if f.layer == 2),
        fires_layer3=0,
        matched_events=m_c,
        missed_events=miss_c,
        median_latency_minutes=_median(lat_c),
        false_positive_proxy=fp_c,
    )

    summary = {
        "station": station,
        "dates": {
            "start": start_date,
            "end": end_date,
        },
        "counts": {
            "events": len(events),
            "obs_rows": len(all_rows),
        },
        "bocpd_enabled": stats_b.__dict__,
        "cusum_only": stats_c.__dict__,
        "delta": {
            "fires": stats_b.total_fires - stats_c.total_fires,
            "false_positive_proxy": stats_b.false_positive_proxy - stats_c.false_positive_proxy,
            "median_latency_minutes": (
                None
                if stats_b.median_latency_minutes is None or stats_c.median_latency_minutes is None
                else round(stats_b.median_latency_minutes - stats_c.median_latency_minutes, 3)
            ),
            "missed_events": stats_b.missed_events - stats_c.missed_events,
        },
    }
    return summary


def _to_markdown(summary: dict) -> str:
    b = summary["bocpd_enabled"]
    c = summary["cusum_only"]
    d = summary["delta"]

    lines = [
        "# Changepoint Replay Comparison",
        "",
        f"- Station: {summary['station']}",
        f"- Rows: {summary['counts']['obs_rows']}",
        f"- Proxy events: {summary['counts']['events']}",
        "",
        "## BOCPD enabled",
        f"- Fires: {b['total_fires']} (L1={b['fires_layer1']}, L2={b['fires_layer2']}, L3={b['fires_layer3']})",
        f"- Matched/missed events: {b['matched_events']}/{b['missed_events']}",
        f"- Median latency (min): {b['median_latency_minutes']}",
        f"- False-positive proxy: {b['false_positive_proxy']}",
        "",
        "## CUSUM only",
        f"- Fires: {c['total_fires']} (L1={c['fires_layer1']}, L2={c['fires_layer2']})",
        f"- Matched/missed events: {c['matched_events']}/{c['missed_events']}",
        f"- Median latency (min): {c['median_latency_minutes']}",
        f"- False-positive proxy: {c['false_positive_proxy']}",
        "",
        "## Delta (BOCPD - CUSUM)",
        f"- Fires: {d['fires']}",
        f"- Missed events: {d['missed_events']}",
        f"- Median latency minutes: {d['median_latency_minutes']}",
        f"- False-positive proxy: {d['false_positive_proxy']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay comparison: BOCPD-enabled vs CUSUM-only change detector")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--out", default="analysis_data/changepoint_compare.json")
    parser.add_argument("--md-out", default="analysis_data/changepoint_compare.md")
    args = parser.parse_args(argv)

    summary = run_compare(
        args.db,
        station=args.station,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_to_markdown(summary), encoding="utf-8")

    print(json.dumps({"ok": True, "out": str(out), "events": summary["counts"]["events"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
