"""Archive/schema readiness audit for weather trading replay and promotion.

Phase mapping:
- T0.1 Canonical archive audit

This module checks whether the database has the core tables/columns needed for
replay-safe inference and compares timestamp freshness by source table.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from engine.replay_context import freshness_status

UTC = timezone.utc


REQUIRED_TABLE_COLUMNS: dict[str, set[str]] = {
    "model_forecasts": {
        "station",
        "forecast_date",
        "model",
        "source",
        "run_time",
        "valid_time",
        "raw_temperature_f",
        "fetch_time_utc",
    },
    "observations": {
        "station",
        "timestamp_utc",
        "temperature_f",
        "dew_point_f",
        "pressure_hpa",
        "sky_cover_pct",
        "source",
    },
    "nearby_observations": {
        "timestamp_utc",
        "air_temp_f",
        "temp_delta_vs_kmia",
    },
    "fawn_observations": {
        "timestamp_utc",
        "air_temp_f",
        "dew_point_f",
    },
    "atmospheric_data": {
        "station",
        "valid_time_utc",
        "cape",
        "precipitable_water_mm",
    },
    "forward_curves": {
        "station",
        "snapshot_time_utc",
        "target_date",
        "valid_hour_utc",
        "hours_ahead",
    },
    "market_snapshots": {
        "ticker",
        "forecast_date",
        "market_type",
        "snapshot_time",
        "best_yes_bid_cents",
        "best_yes_ask_cents",
        "best_no_bid_cents",
        "best_no_ask_cents",
    },
    "active_brackets": {
        "ticker",
        "target_date",
        "market_type",
        "settlement_floor",
        "settlement_ceil",
    },
    "bracket_estimates": {
        "station",
        "target_date",
        "ticker",
        "probability",
        "timestamp_utc",
    },
    "market_settlements": {
        "ticker",
        "station",
        "forecast_date",
        "market_type",
        "winning_side",
    },
    "event_settlements": {
        "station",
        "settlement_date",
        "market_type",
        "actual_value_f",
    },
}

TIMESTAMP_CANDIDATES: dict[str, tuple[str, ...]] = {
    "model_forecasts": ("fetch_time_utc", "run_time", "created_at"),
    "observations": ("timestamp_utc", "created_at"),
    "nearby_observations": ("timestamp_utc", "created_at"),
    "fawn_observations": ("timestamp_utc", "created_at"),
    "atmospheric_data": ("valid_time_utc", "fetch_time_utc", "created_at"),
    "forward_curves": ("snapshot_time_utc", "created_at"),
    "market_snapshots": ("snapshot_time", "created_at"),
    "active_brackets": ("updated_at",),
    "bracket_estimates": ("timestamp_utc",),
    "market_settlements": ("settled_at", "created_at"),
    "event_settlements": ("received_at", "created_at"),
}

TABLE_FRESHNESS_MODE: dict[str, str] = {
    "model_forecasts": "realtime",
    "observations": "realtime",
    "nearby_observations": "realtime",
    "fawn_observations": "realtime",
    "market_snapshots": "realtime",
    "active_brackets": "realtime",
    "bracket_estimates": "realtime",
    "atmospheric_data": "forecast_validity",
    "forward_curves": "forecast_validity",
    "market_settlements": "historical",
    "event_settlements": "historical",
}

STALE_MINUTES_BY_TABLE: dict[str, int] = {
    "model_forecasts": 120,
    "observations": 20,
    "nearby_observations": 20,
    "fawn_observations": 30,
    "market_snapshots": 5,
    "active_brackets": 10,
    "bracket_estimates": 10,
    "atmospheric_data": 180,
    "forward_curves": 180,
    "market_settlements": 7 * 24 * 60,
    "event_settlements": 7 * 24 * 60,
}


@dataclass
class TableAudit:
    table: str
    present: bool
    row_count: int
    required_columns: list[str]
    missing_columns: list[str]
    timestamp_column: str | None
    latest_timestamp: str | None
    freshness_mode: str | None
    freshness_status: str | None
    freshness_minutes: float | None


@dataclass
class ArchiveAuditReport:
    db_path: str
    station: str | None
    generated_at_utc: str
    total_required_tables: int
    missing_tables: list[str]
    table_audits: list[TableAudit]

    @property
    def is_replay_ready(self) -> bool:
        if self.missing_tables:
            return False
        return all(not ta.missing_columns for ta in self.table_audits if ta.present)



def _classify_freshness(
    *,
    table: str,
    now_utc: datetime,
    latest_ts: str | None,
) -> tuple[str | None, str | None, float | None]:
    mode = TABLE_FRESHNESS_MODE.get(table, "realtime")
    stale_after = STALE_MINUTES_BY_TABLE.get(table, 30)

    status, age = freshness_status(now_utc, latest_ts, stale_after_minutes=stale_after)

    if mode == "forecast_validity" and status == "future":
        return mode, "future_expected", age
    if mode == "historical" and status not in {"missing", "unparseable"}:
        return mode, "historical", age

    return mode, status, age


def _list_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(r[0]) for r in rows}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def _latest_ts_and_col(
    conn: sqlite3.Connection,
    table: str,
    columns: set[str],
) -> tuple[str | None, str | None]:
    for candidate in TIMESTAMP_CANDIDATES.get(table, ()):  # first viable column wins
        if candidate not in columns:
            continue
        row = conn.execute(f"SELECT MAX({candidate}) FROM {table}").fetchone()
        latest = row[0] if row else None
        if latest is not None:
            return str(latest), candidate
        return None, candidate
    return None, None


def audit_connection(
    conn: sqlite3.Connection,
    *,
    db_path: str,
    station: str | None = None,
    now_utc: datetime | None = None,
) -> ArchiveAuditReport:
    now = now_utc or datetime.now(UTC)
    tables_present = _list_tables(conn)

    missing_tables = sorted(set(REQUIRED_TABLE_COLUMNS) - tables_present)
    table_audits: list[TableAudit] = []

    for table in sorted(REQUIRED_TABLE_COLUMNS):
        required_cols = sorted(REQUIRED_TABLE_COLUMNS[table])
        if table not in tables_present:
            table_audits.append(
                TableAudit(
                    table=table,
                    present=False,
                    row_count=0,
                    required_columns=required_cols,
                    missing_columns=required_cols,
                    timestamp_column=None,
                    latest_timestamp=None,
                    freshness_mode=None,
                    freshness_status=None,
                    freshness_minutes=None,
                )
            )
            continue

        cols = _table_columns(conn, table)
        missing_cols = sorted(set(required_cols) - cols)

        row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        latest_ts, ts_col = _latest_ts_and_col(conn, table, cols)

        freshness_mode, freshness_state, freshness = _classify_freshness(
            table=table,
            now_utc=now,
            latest_ts=latest_ts,
        )

        table_audits.append(
            TableAudit(
                table=table,
                present=True,
                row_count=row_count,
                required_columns=required_cols,
                missing_columns=missing_cols,
                timestamp_column=ts_col,
                latest_timestamp=latest_ts,
                freshness_mode=freshness_mode,
                freshness_status=freshness_state,
                freshness_minutes=freshness,
            )
        )

    return ArchiveAuditReport(
        db_path=db_path,
        station=station,
        generated_at_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_required_tables=len(REQUIRED_TABLE_COLUMNS),
        missing_tables=missing_tables,
        table_audits=table_audits,
    )


def run_archive_audit(
    db_path: str | Path,
    *,
    station: str | None = None,
    now_utc: datetime | None = None,
) -> ArchiveAuditReport:
    conn = sqlite3.connect(str(db_path))
    try:
        return audit_connection(
            conn,
            db_path=str(db_path),
            station=station,
            now_utc=now_utc,
        )
    finally:
        conn.close()


def report_to_markdown(report: ArchiveAuditReport) -> str:
    lines = []
    lines.append("# Archive Audit Report")
    lines.append("")
    lines.append(f"- DB: `{report.db_path}`")
    lines.append(f"- Generated (UTC): `{report.generated_at_utc}`")
    if report.station:
        lines.append(f"- Station: `{report.station}`")
    lines.append(f"- Required tables: `{report.total_required_tables}`")
    lines.append(f"- Missing tables: `{len(report.missing_tables)}`")
    lines.append(f"- Replay-ready (schema): `{report.is_replay_ready}`")
    lines.append("")

    if report.missing_tables:
        lines.append("## Missing required tables")
        for t in report.missing_tables:
            lines.append(f"- `{t}`")
        lines.append("")

    lines.append("## Table details")
    for ta in report.table_audits:
        lines.append("")
        lines.append(f"### `{ta.table}`")
        lines.append(f"- present: `{ta.present}`")
        lines.append(f"- rows: `{ta.row_count}`")
        lines.append(f"- missing required columns: `{len(ta.missing_columns)}`")
        if ta.missing_columns:
            lines.append(f"- missing list: `{', '.join(ta.missing_columns)}`")
        lines.append(f"- timestamp column: `{ta.timestamp_column}`")
        lines.append(f"- latest timestamp: `{ta.latest_timestamp}`")
        lines.append(f"- freshness mode: `{ta.freshness_mode}`")
        lines.append(f"- freshness status: `{ta.freshness_status}`")
        lines.append(f"- freshness minutes: `{ta.freshness_minutes}`")

    return "\n".join(lines) + "\n"


def _json_default(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit archive schema/readiness for replay")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--out-md", help="Write markdown report to this path")
    parser.add_argument("--out-json", help="Write JSON report to this path")
    args = parser.parse_args(argv)

    report = run_archive_audit(args.db, station=args.station)
    md = report_to_markdown(report)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "db_path": report.db_path,
            "station": report.station,
            "generated_at_utc": report.generated_at_utc,
            "total_required_tables": report.total_required_tables,
            "missing_tables": report.missing_tables,
            "is_replay_ready": report.is_replay_ready,
            "table_audits": [asdict(t) for t in report.table_audits],
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
