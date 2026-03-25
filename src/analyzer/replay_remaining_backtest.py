"""Historical replay backtest for the remaining-only orchestrator path.

Replays inference at historical timestamps (forward-curve snapshots) and
simulates multi-entry-per-ticker trades using the shared portfolio middleman.

Outputs a compact JSON summary with:
- calibration (Brier/log-loss), reliability, and sharpness diagnostics
- time-to-settlement (TTL) calibration cuts
- simulated trade count + expected value + realized PnL to settlement
- per-day breakdown and worst-day diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.orchestrator import InferenceRuntimeState, run_inference_cycle

_WEATHER_TRADER_SRC = Path(__file__).resolve().parents[3] / "weather-trader" / "src"
if str(_WEATHER_TRADER_SRC) not in sys.path:
    sys.path.append(str(_WEATHER_TRADER_SRC))

from engine.replay_context import parse_utc
from execution.portfolio import PortfolioMiddleman, SettledTradeRecord
from execution.trader_policy import kalshi_fee_cents

UTC = timezone.utc


@dataclass
class ReplayTrade:
    target_date: str
    ticker: str
    market_type: str
    primary_regime: str
    side: str
    contracts: int
    entry_time_utc: str
    entry_price_cents: float
    model_probability_yes: float
    adjusted_probability_side: float
    regime_confidence: float
    edge_cents: float
    expected_value_cents: float
    winning_side: str
    realized_pnl_cents: float


def _clamp_prob(p: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, p))


def _settled_dates(
    conn: sqlite3.Connection,
    station: str,
    start_date: str | None,
    end_date: str | None,
    explicit_dates: list[str] | None,
) -> list[str]:
    if explicit_dates:
        return sorted(set(explicit_dates))

    sql = """SELECT DISTINCT forecast_date
             FROM market_settlements
             WHERE station = ? AND forecast_date IS NOT NULL"""
    params: list[object] = [station]
    if start_date:
        sql += " AND forecast_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND forecast_date <= ?"
        params.append(end_date)
    sql += " ORDER BY forecast_date"

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [str(r[0]) for r in rows]


def _schedule_for_day(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    step_minutes: int,
    max_cycles: int | None,
) -> list[datetime]:
    rows = conn.execute(
        """SELECT DISTINCT snapshot_time_utc
           FROM forward_curves
           WHERE station = ? AND target_date = ?
           ORDER BY snapshot_time_utc""",
        (station, target_date),
    ).fetchall()

    if not rows:
        # Fallback: market snapshots if forward_curves are missing.
        rows = conn.execute(
            """SELECT DISTINCT snapshot_time
               FROM market_snapshots
               WHERE forecast_date = ?
               ORDER BY snapshot_time""",
            (target_date,),
        ).fetchall()

    times: list[datetime] = []
    for r in rows:
        if not r[0]:
            continue
        dt = parse_utc(str(r[0]))
        if dt is not None:
            times.append(dt)
    if not times:
        return []

    keep: list[datetime] = []
    min_gap = step_minutes * 60
    last_kept: datetime | None = None
    for ts in times:
        if last_kept is None or (ts - last_kept).total_seconds() >= min_gap:
            keep.append(ts)
            last_kept = ts
            if max_cycles is not None and len(keep) >= max_cycles:
                break

    return keep


def _quotes_as_of(conn: sqlite3.Connection, target_date: str, as_of: datetime) -> dict[str, sqlite3.Row]:
    ts = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = conn.execute(
        """SELECT ms.*
           FROM market_snapshots ms
           JOIN (
               SELECT ticker, MAX(id) AS max_id
               FROM market_snapshots
               WHERE forecast_date = ? AND snapshot_time <= ?
               GROUP BY ticker
           ) latest ON latest.max_id = ms.id""",
        (target_date, ts),
    ).fetchall()
    return {str(r["ticker"]): r for r in rows}


def _settlement_map(conn: sqlite3.Connection, station: str, target_date: str) -> dict[str, str]:
    rows = conn.execute(
        """SELECT ticker, winning_side
           FROM market_settlements
           WHERE station = ? AND forecast_date = ?""",
        (station, target_date),
    ).fetchall()
    return {str(r["ticker"]): str(r["winning_side"]).lower() for r in rows if r["winning_side"]}


def _brier(prob: float, y: int) -> float:
    return (prob - float(y)) ** 2


def _logloss(prob: float, y: int) -> float:
    p = _clamp_prob(prob)
    return -(math.log(p) if y == 1 else math.log(1.0 - p))


def _pinball_loss(q: float, forecast: float, actual: float) -> float:
    err = float(actual) - float(forecast)
    return q * err if err >= 0 else (1.0 - q) * (-err)


def _approx_crps_from_quantiles(quantiles: dict[float, float], actual: float) -> float | None:
    if not quantiles:
        return None
    ordered = sorted((float(q), float(v)) for q, v in quantiles.items())
    if not ordered:
        return None
    score = sum(2.0 * _pinball_loss(q, v, actual) for q, v in ordered) / len(ordered)
    return float(score)


def _rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def _remaining_quantiles(
    market_type: str,
    quantiles: dict[float, float],
    *,
    running_high_f: float | None,
    running_low_f: float | None,
) -> dict[float, float]:
    if not quantiles:
        return {}

    if market_type == "high":
        if running_high_f is None:
            return {}
        return {
            float(q): round(max(0.0, float(v) - float(running_high_f)), 6)
            for q, v in quantiles.items()
        }

    if running_low_f is None:
        return {}

    out: dict[float, float] = {}
    normalized = {round(float(q), 6): float(v) for q, v in quantiles.items()}
    for q in normalized:
        mirror_q = round(1.0 - q, 6)
        mirror_v = normalized.get(mirror_q)
        if mirror_v is None:
            continue
        out[q] = round(max(0.0, float(running_low_f) - mirror_v), 6)
    return out


def _confidence_bucket(confidence: float) -> str:
    c = max(0.0, min(1.0, float(confidence)))
    if c < 0.55:
        return "low"
    if c < 0.75:
        return "medium"
    return "high"


def _reliability_bins(prob_label_pairs: list[tuple[float, int]], bins: int = 10) -> list[dict[str, float | int]]:
    if not prob_label_pairs:
        return []

    out: list[dict[str, float | int]] = []
    width = 1.0 / max(1, int(bins))
    for i in range(int(bins)):
        lo = i * width
        hi = 1.0 if i == bins - 1 else (i + 1) * width

        rows = [
            (p, y)
            for (p, y) in prob_label_pairs
            if (p >= lo and (p < hi if i < bins - 1 else p <= hi))
        ]
        if not rows:
            continue

        n = len(rows)
        mean_p = sum(p for p, _ in rows) / n
        yes_rate = sum(y for _, y in rows) / n
        out.append(
            {
                "bin": i,
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "n": n,
                "mean_prob": round(mean_p, 6),
                "empirical_yes": round(yes_rate, 6),
                "calibration_gap": round(mean_p - yes_rate, 6),
            }
        )
    return out


def _climate_day_end_utc(target_date: str) -> datetime:
    d = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)
    return datetime(d.year, d.month, d.day, 5, 0, 0, tzinfo=UTC)


def _ttl_bucket(hours_to_settlement: float) -> str:
    h = max(0.0, float(hours_to_settlement))
    if h < 3:
        return "0-3h"
    if h < 6:
        return "3-6h"
    if h < 12:
        return "6-12h"
    if h < 24:
        return "12-24h"
    return "24h+"


def _climate_day_start_utc(target_date: str) -> datetime:
    return _climate_day_end_utc(target_date) - timedelta(days=1)


def _event_actuals(conn: sqlite3.Connection, station: str, target_date: str) -> dict[str, float]:
    rows = conn.execute(
        """SELECT market_type, actual_value_f
           FROM event_settlements
           WHERE station = ? AND settlement_date = ? AND actual_value_f IS NOT NULL""",
        (station, target_date),
    ).fetchall()
    out: dict[str, float] = {}
    for row in rows:
        market_type = str(row["market_type"] or "").lower().strip()
        if market_type in {"high", "low"}:
            out[market_type] = float(row["actual_value_f"])
    return out


def _running_extremes_as_of(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    as_of: datetime,
) -> tuple[float | None, float | None]:
    start = _climate_day_start_utc(target_date).strftime("%Y-%m-%dT%H:%M:%SZ")
    effective_end = min(_climate_day_end_utc(target_date), as_of).strftime("%Y-%m-%dT%H:%M:%SZ")

    row = conn.execute(
        """SELECT MAX(wethr_high_nws_f), MIN(wethr_low_nws_f)
           FROM observations
           WHERE station = ?
             AND timestamp_utc >= ? AND timestamp_utc < ?
             AND wethr_high_nws_f IS NOT NULL""",
        (station, start, effective_end),
    ).fetchone()
    if row and row[0] is not None:
        return (float(row[0]), float(row[1]) if row[1] is not None else None)

    row = conn.execute(
        """SELECT MAX(temperature_f), MIN(temperature_f)
           FROM observations
           WHERE station = ?
             AND timestamp_utc >= ? AND timestamp_utc < ?
             AND temperature_f IS NOT NULL""",
        (station, start, effective_end),
    ).fetchone()
    if not row:
        return (None, None)
    return (
        float(row[0]) if row[0] is not None else None,
        float(row[1]) if row[1] is not None else None,
    )


def _history_records_for_window(
    station: str,
    trades: list[ReplayTrade],
    *,
    day_start_utc: datetime,
    adaptive_lookback_days: int,
) -> list[SettledTradeRecord]:
    window_start = day_start_utc.timestamp() - max(1, int(adaptive_lookback_days)) * 86400
    out: list[SettledTradeRecord] = []
    for t in trades:
        dt = datetime.strptime(t.target_date, "%Y-%m-%d").replace(tzinfo=UTC)
        ts = dt.timestamp()
        if ts >= day_start_utc.timestamp() or ts < window_start:
            continue
        out.append(
            SettledTradeRecord(
                station=station,
                side=t.side,
                winning_side=t.winning_side,
                estimated_probability=float(t.adjusted_probability_side),
                realized_pnl_cents=float(t.realized_pnl_cents),
                contracts=max(1, int(t.contracts)),
                settled_at_utc=dt,
            )
        )
    return out



def run_replay(
    db_path: str | Path,
    station: str,
    dates: list[str],
    step_minutes: int,
    max_cycles_per_day: int | None,
    max_trades_per_ticker: int,
    min_reentry_minutes: int,
    adaptive_lookback_days: int,
) -> tuple[dict, list[ReplayTrade]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    portfolio = PortfolioMiddleman(
        adaptive_tuning=True,
        adaptive_lookback_hours=max(24, int(adaptive_lookback_days) * 24),
        adaptive_min_settled_trades=8,
        default_max_open_trades_per_ticker=max_trades_per_ticker,
        default_min_minutes_between_entries=min_reentry_minutes,
    )

    all_brier: list[float] = []
    all_logloss: list[float] = []
    all_trade_brier: list[float] = []
    prob_label_pairs: list[tuple[float, int]] = []

    ttl_prob_label_pairs: dict[str, list[tuple[float, int]]] = {
        "0-3h": [],
        "3-6h": [],
        "6-12h": [],
        "12-24h": [],
        "24h+": [],
    }

    day_metrics: dict[str, dict[str, float | int]] = {}

    remaining_errors: dict[str, list[float]] = {"high": [], "low": []}
    remaining_crps: dict[str, list[float]] = {"high": [], "low": []}
    remaining_ttl: dict[str, dict[str, list[float]]] = {
        bucket: {
            "high_errors": [],
            "low_errors": [],
            "high_crps": [],
            "low_crps": [],
        }
        for bucket in ("0-3h", "3-6h", "6-12h", "12-24h", "24h+")
    }
    regime_remaining: dict[tuple[str, str], dict[str, list[float] | int]] = {}

    trades: list[ReplayTrade] = []
    total_cycles = 0
    total_estimates = 0

    try:
        for target_date in dates:
            outcomes = _settlement_map(conn, station, target_date)
            if not outcomes:
                continue

            schedule = _schedule_for_day(conn, station, target_date, step_minutes, max_cycles_per_day)
            if not schedule:
                continue

            target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=UTC)
            history_records = _history_records_for_window(
                station,
                trades,
                day_start_utc=target_dt,
                adaptive_lookback_days=adaptive_lookback_days,
            )
            portfolio.refresh(
                stations=[station],
                settled_rows=history_records,
                now_utc=target_dt,
            )

            runtime_state = InferenceRuntimeState()
            trades_per_ticker: dict[str, int] = {}
            last_trade_time: dict[str, datetime] = {}

            day_metrics.setdefault(
                target_date,
                {
                    "estimates": 0,
                    "brier_sum": 0.0,
                    "logloss_sum": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "contracts": 0,
                    "expected_value_cents_sum": 0.0,
                    "realized_pnl_cents_sum": 0.0,
                    "remaining_high_abs_sum": 0.0,
                    "remaining_high_sq_sum": 0.0,
                    "remaining_high_n": 0,
                    "remaining_low_abs_sum": 0.0,
                    "remaining_low_sq_sum": 0.0,
                    "remaining_low_n": 0,
                },
            )

            settlement_end = _climate_day_end_utc(target_date)
            event_actuals = _event_actuals(conn, station, target_date)

            for ts in schedule:
                result = run_inference_cycle(
                    db_path,
                    target_date=target_date,
                    station=station,
                    runtime_state=runtime_state,
                    eval_time_utc=ts,
                )
                total_cycles += 1
                total_estimates += len(result.estimates)

                # Calibration over all replayed probabilities (only settled tickers).
                for est in result.estimates:
                    winner = outcomes.get(est.ticker)
                    if winner is None:
                        continue
                    y = 1 if winner == "yes" else 0
                    p_yes = float(est.model_probability)
                    brier = _brier(p_yes, y)
                    logloss = _logloss(p_yes, y)

                    all_brier.append(brier)
                    all_logloss.append(logloss)
                    prob_label_pairs.append((p_yes, y))

                    day_entry = day_metrics[target_date]
                    day_entry["estimates"] = int(day_entry["estimates"]) + 1
                    day_entry["brier_sum"] = float(day_entry["brier_sum"]) + brier
                    day_entry["logloss_sum"] = float(day_entry["logloss_sum"]) + logloss

                    ttl_hours = max(0.0, (settlement_end - ts).total_seconds() / 3600.0)
                    ttl_prob_label_pairs[_ttl_bucket(ttl_hours)].append((p_yes, y))

                running_high_f, running_low_f = _running_extremes_as_of(conn, station, target_date, ts)
                ttl_bucket = _ttl_bucket(max(0.0, (settlement_end - ts).total_seconds() / 3600.0))

                market_eval_rows = [
                    ("high", result.high_belief, result.high_regime, event_actuals.get("high"), running_high_f, running_low_f),
                    ("low", result.low_belief, result.low_regime, event_actuals.get("low"), running_high_f, running_low_f),
                ]
                for market_type, belief, regime_state, actual_final, run_hi, run_lo in market_eval_rows:
                    if belief is None or actual_final is None:
                        continue
                    dist = belief.distribution
                    mu = dist.mu
                    if mu is None:
                        continue

                    if market_type == "high":
                        if run_hi is None:
                            continue
                        actual_remaining = max(0.0, float(actual_final) - float(run_hi))
                        pred_remaining = max(0.0, float(mu) - float(run_hi))
                    else:
                        if run_lo is None:
                            continue
                        actual_remaining = max(0.0, float(run_lo) - float(actual_final))
                        pred_remaining = max(0.0, float(run_lo) - float(mu))

                    err = pred_remaining - actual_remaining
                    remaining_errors[market_type].append(err)
                    day_entry = day_metrics[target_date]
                    if market_type == "high":
                        day_entry["remaining_high_abs_sum"] = float(day_entry["remaining_high_abs_sum"]) + abs(err)
                        day_entry["remaining_high_sq_sum"] = float(day_entry["remaining_high_sq_sum"]) + err * err
                        day_entry["remaining_high_n"] = int(day_entry["remaining_high_n"]) + 1
                    else:
                        day_entry["remaining_low_abs_sum"] = float(day_entry["remaining_low_abs_sum"]) + abs(err)
                        day_entry["remaining_low_sq_sum"] = float(day_entry["remaining_low_sq_sum"]) + err * err
                        day_entry["remaining_low_n"] = int(day_entry["remaining_low_n"]) + 1

                    rem_quantiles = _remaining_quantiles(
                        market_type,
                        dist.quantiles,
                        running_high_f=run_hi,
                        running_low_f=run_lo,
                    )
                    crps_val = _approx_crps_from_quantiles(rem_quantiles, actual_remaining)
                    if crps_val is not None:
                        remaining_crps[market_type].append(crps_val)
                        remaining_ttl[ttl_bucket][f"{market_type}_crps"].append(crps_val)

                    remaining_ttl[ttl_bucket][f"{market_type}_errors"].append(err)

                    regime_name = regime_state.primary_regime if regime_state is not None else "unknown"
                    regime_entry = regime_remaining.setdefault(
                        (market_type, regime_name),
                        {"n": 0, "errors": [], "crps": []},
                    )
                    regime_entry["n"] = int(regime_entry["n"]) + 1
                    cast_errors = regime_entry["errors"]
                    if isinstance(cast_errors, list):
                        cast_errors.append(err)
                    if crps_val is not None:
                        cast_crps = regime_entry["crps"]
                        if isinstance(cast_crps, list):
                            cast_crps.append(crps_val)

                # Entry simulation: allow multiple entries per ticker/day (policy bounded).
                quotes = _quotes_as_of(conn, target_date, ts)
                for est in result.estimates:
                    prior_entries = trades_per_ticker.get(est.ticker, 0)
                    last_entry = last_trade_time.get(est.ticker)

                    winner = outcomes.get(est.ticker)
                    if winner is None:
                        continue

                    q = quotes.get(est.ticker)
                    if q is None:
                        continue

                    regime_conf = float(est.regime_confidence if est.regime_confidence is not None else 0.5)
                    rec = portfolio.recommend_trade(
                        station=station,
                        now_utc=ts,
                        open_count_for_ticker=prior_entries,
                        last_entry_utc=last_entry,
                        prob_yes=float(est.model_probability),
                        regime_confidence=regime_conf,
                        yes_ask_cents=(float(q["best_yes_ask_cents"]) if q["best_yes_ask_cents"] is not None else None),
                        yes_ask_qty=int(q["yes_ask_qty"] or 0),
                        no_ask_cents=(float(q["best_no_ask_cents"]) if q["best_no_ask_cents"] is not None else None),
                        no_ask_qty=int(q["no_ask_qty"] or 0),
                    )
                    if rec is None:
                        continue

                    payout = 100.0 if rec.side == winner else 0.0
                    fee = kalshi_fee_cents(rec.ask_price_cents)
                    pnl_per_contract = payout - rec.ask_price_cents - fee
                    pnl_total = round(pnl_per_contract * rec.contracts, 3)

                    y_trade = 1 if rec.side == winner else 0
                    all_trade_brier.append(_brier(float(rec.probability), y_trade))

                    regime_name = (
                        result.high_regime.primary_regime
                        if est.market_type == "high" and result.high_regime is not None
                        else result.low_regime.primary_regime
                        if est.market_type == "low" and result.low_regime is not None
                        else "unknown"
                    )

                    trades.append(
                        ReplayTrade(
                            target_date=target_date,
                            ticker=est.ticker,
                            market_type=est.market_type,
                            primary_regime=regime_name,
                            side=rec.side,
                            contracts=int(rec.contracts),
                            entry_time_utc=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            entry_price_cents=float(rec.ask_price_cents),
                            model_probability_yes=float(est.model_probability),
                            adjusted_probability_side=float(rec.probability),
                            regime_confidence=regime_conf,
                            edge_cents=float(rec.edge_cents),
                            expected_value_cents=float(rec.expected_value_cents),
                            winning_side=winner,
                            realized_pnl_cents=pnl_total,
                        )
                    )

                    day_entry = day_metrics[target_date]
                    day_entry["trades"] = int(day_entry["trades"]) + 1
                    day_entry["wins"] = int(day_entry["wins"]) + (1 if rec.side == winner else 0)
                    day_entry["contracts"] = int(day_entry["contracts"]) + int(rec.contracts)
                    day_entry["expected_value_cents_sum"] = (
                        float(day_entry["expected_value_cents_sum"]) + float(rec.expected_value_cents)
                    )
                    day_entry["realized_pnl_cents_sum"] = (
                        float(day_entry["realized_pnl_cents_sum"]) + float(pnl_total)
                    )

                    trades_per_ticker[est.ticker] = prior_entries + 1
                    last_trade_time[est.ticker] = ts

        trade_count = len(trades)
        total_contracts = sum(t.contracts for t in trades)
        total_pnl_cents = round(sum(t.realized_pnl_cents for t in trades), 3)
        total_expected_value_cents = round(sum(t.expected_value_cents for t in trades), 3)
        wins = sum(1 for t in trades if t.side == t.winning_side)

        ttl_cuts: dict[str, dict[str, float | int | list[dict[str, float | int]] | dict[str, float | int | None]]] = {}
        for bucket, pairs in ttl_prob_label_pairs.items():
            has_remaining = any(remaining_ttl[bucket][k] for k in remaining_ttl[bucket])
            if not pairs and not has_remaining:
                continue

            row: dict[str, float | int | list[dict[str, float | int]] | dict[str, float | int | None]] = {}
            if pairs:
                brier_mean = sum(_brier(p, y) for p, y in pairs) / len(pairs)
                logloss_mean = sum(_logloss(p, y) for p, y in pairs) / len(pairs)
                row.update(
                    {
                        "n": len(pairs),
                        "brier": round(brier_mean, 6),
                        "logloss": round(logloss_mean, 6),
                        "reliability_bins": _reliability_bins(pairs, bins=5),
                    }
                )

            rem_bucket: dict[str, float | int | None] = {}
            for market_type in ("high", "low"):
                errs = remaining_ttl[bucket][f"{market_type}_errors"]
                crps_vals = remaining_ttl[bucket][f"{market_type}_crps"]
                if not errs and not crps_vals:
                    continue
                rem_bucket[f"remaining_{market_type}_n"] = len(errs)
                rem_bucket[f"remaining_{market_type}_mae"] = round(sum(abs(e) for e in errs) / len(errs), 6) if errs else None
                rem_bucket[f"remaining_{market_type}_rmse"] = round(_rmse(errs) or 0.0, 6) if errs else None
                rem_bucket[f"remaining_{market_type}_crps"] = round(sum(crps_vals) / len(crps_vals), 6) if crps_vals else None
            if rem_bucket:
                row["remaining_targets"] = rem_bucket
            ttl_cuts[bucket] = row

        regime_cuts: list[dict[str, float | int | str | None]] = []
        for (market_type, regime_name), payload in sorted(regime_remaining.items()):
            errs = payload.get("errors") if isinstance(payload, dict) else []
            crps_vals = payload.get("crps") if isinstance(payload, dict) else []
            errs = errs if isinstance(errs, list) else []
            crps_vals = crps_vals if isinstance(crps_vals, list) else []
            regime_cuts.append(
                {
                    "market_type": market_type,
                    "regime": regime_name,
                    "n": int(payload.get("n", len(errs))) if isinstance(payload, dict) else len(errs),
                    "remaining_mae": round(sum(abs(e) for e in errs) / len(errs), 6) if errs else None,
                    "remaining_rmse": round(_rmse(errs) or 0.0, 6) if errs else None,
                    "remaining_crps": round(sum(crps_vals) / len(crps_vals), 6) if crps_vals else None,
                }
            )

        trade_quality_by_market: list[dict[str, float | int | str | None]] = []
        for market_type in sorted({t.market_type for t in trades}):
            rows = [t for t in trades if t.market_type == market_type]
            if not rows:
                continue
            wins_m = sum(1 for t in rows if t.side == t.winning_side)
            trade_quality_by_market.append(
                {
                    "market_type": market_type,
                    "n": len(rows),
                    "win_rate": round(wins_m / len(rows), 6),
                    "avg_edge_cents": round(sum(t.edge_cents for t in rows) / len(rows), 6),
                    "avg_expected_value_cents": round(sum(t.expected_value_cents for t in rows) / len(rows), 6),
                    "avg_realized_pnl_cents": round(sum(t.realized_pnl_cents for t in rows) / len(rows), 6),
                }
            )

        trade_quality_by_confidence: list[dict[str, float | int | str | None]] = []
        for bucket in ("low", "medium", "high"):
            rows = [t for t in trades if _confidence_bucket(t.regime_confidence) == bucket]
            if not rows:
                continue
            wins_b = sum(1 for t in rows if t.side == t.winning_side)
            trade_quality_by_confidence.append(
                {
                    "confidence_bucket": bucket,
                    "n": len(rows),
                    "win_rate": round(wins_b / len(rows), 6),
                    "avg_regime_confidence": round(sum(t.regime_confidence for t in rows) / len(rows), 6),
                    "avg_edge_cents": round(sum(t.edge_cents for t in rows) / len(rows), 6),
                    "avg_expected_value_cents": round(sum(t.expected_value_cents for t in rows) / len(rows), 6),
                    "avg_realized_pnl_cents": round(sum(t.realized_pnl_cents for t in rows) / len(rows), 6),
                }
            )

        trade_quality_by_regime: list[dict[str, float | int | str | None]] = []
        for regime_name in sorted({t.primary_regime for t in trades}):
            rows = [t for t in trades if t.primary_regime == regime_name]
            if not rows:
                continue
            wins_r = sum(1 for t in rows if t.side == t.winning_side)
            trade_quality_by_regime.append(
                {
                    "regime": regime_name,
                    "n": len(rows),
                    "win_rate": round(wins_r / len(rows), 6),
                    "avg_edge_cents": round(sum(t.edge_cents for t in rows) / len(rows), 6),
                    "avg_expected_value_cents": round(sum(t.expected_value_cents for t in rows) / len(rows), 6),
                    "avg_realized_pnl_cents": round(sum(t.realized_pnl_cents for t in rows) / len(rows), 6),
                }
            )

        day_breakdown: list[dict[str, float | int | str | None]] = []
        for d in sorted(day_metrics):
            m = day_metrics[d]
            estimates_n = int(m["estimates"])
            trades_n = int(m["trades"])
            wins_n = int(m["wins"])
            hi_n = int(m["remaining_high_n"])
            lo_n = int(m["remaining_low_n"])
            day_breakdown.append(
                {
                    "date": d,
                    "estimates": estimates_n,
                    "brier": (
                        round(float(m["brier_sum"]) / estimates_n, 6)
                        if estimates_n > 0
                        else None
                    ),
                    "logloss": (
                        round(float(m["logloss_sum"]) / estimates_n, 6)
                        if estimates_n > 0
                        else None
                    ),
                    "remaining_high_mae": (
                        round(float(m["remaining_high_abs_sum"]) / hi_n, 6)
                        if hi_n > 0
                        else None
                    ),
                    "remaining_high_rmse": (
                        round(math.sqrt(float(m["remaining_high_sq_sum"]) / hi_n), 6)
                        if hi_n > 0
                        else None
                    ),
                    "remaining_low_mae": (
                        round(float(m["remaining_low_abs_sum"]) / lo_n, 6)
                        if lo_n > 0
                        else None
                    ),
                    "remaining_low_rmse": (
                        round(math.sqrt(float(m["remaining_low_sq_sum"]) / lo_n), 6)
                        if lo_n > 0
                        else None
                    ),
                    "trades": trades_n,
                    "wins": wins_n,
                    "win_rate": round(wins_n / trades_n, 6) if trades_n > 0 else None,
                    "contracts": int(m["contracts"]),
                    "expected_value_cents": round(float(m["expected_value_cents_sum"]), 3),
                    "realized_pnl_cents": round(float(m["realized_pnl_cents_sum"]), 3),
                }
            )

        worst_brier_day = None
        brier_candidates = [d for d in day_breakdown if d.get("brier") is not None]
        if brier_candidates:
            worst_brier_day = max(brier_candidates, key=lambda x: float(x["brier"]))

        worst_pnl_day = None
        if day_breakdown:
            worst_pnl_day = min(day_breakdown, key=lambda x: float(x["realized_pnl_cents"]))

        summary = {
            "station": station,
            "dates": dates,
            "days": len(dates),
            "cycles": total_cycles,
            "estimates": total_estimates,
            "brier_all_estimates": round(sum(all_brier) / len(all_brier), 6) if all_brier else None,
            "logloss_all_estimates": round(sum(all_logloss) / len(all_logloss), 6) if all_logloss else None,
            "trade_brier": round(sum(all_trade_brier) / len(all_trade_brier), 6) if all_trade_brier else None,
            "sharpness_mean_abs_p_minus_0_5": (
                round(sum(abs(p - 0.5) for p, _ in prob_label_pairs) / len(prob_label_pairs), 6)
                if prob_label_pairs
                else None
            ),
            "remaining_target_metrics": {
                "high": {
                    "n": len(remaining_errors["high"]),
                    "mae": round(sum(abs(e) for e in remaining_errors["high"]) / len(remaining_errors["high"]), 6) if remaining_errors["high"] else None,
                    "rmse": round(_rmse(remaining_errors["high"]) or 0.0, 6) if remaining_errors["high"] else None,
                    "crps": round(sum(remaining_crps["high"]) / len(remaining_crps["high"]), 6) if remaining_crps["high"] else None,
                },
                "low": {
                    "n": len(remaining_errors["low"]),
                    "mae": round(sum(abs(e) for e in remaining_errors["low"]) / len(remaining_errors["low"]), 6) if remaining_errors["low"] else None,
                    "rmse": round(_rmse(remaining_errors["low"]) or 0.0, 6) if remaining_errors["low"] else None,
                    "crps": round(sum(remaining_crps["low"]) / len(remaining_crps["low"]), 6) if remaining_crps["low"] else None,
                },
                "crps_method": "quantile-grid approximation over remaining-target transformed predictive distribution",
            },
            "reliability_bins": _reliability_bins(prob_label_pairs, bins=10),
            "ttl_cuts": ttl_cuts,
            "regime_cuts": regime_cuts,
            "trade_quality_cuts": {
                "by_market_type": trade_quality_by_market,
                "by_regime": trade_quality_by_regime,
                "by_confidence_bucket": trade_quality_by_confidence,
            },
            "day_breakdown": day_breakdown,
            "worst_day_diagnostics": {
                "highest_brier_day": worst_brier_day,
                "lowest_realized_pnl_day": worst_pnl_day,
            },
            "trades": trade_count,
            "wins": wins,
            "win_rate": round(wins / trade_count, 4) if trade_count else None,
            "contracts": total_contracts,
            "avg_contracts": round(total_contracts / trade_count, 3) if trade_count else None,
            "pnl_cents": total_pnl_cents,
            "pnl_dollars": round(total_pnl_cents / 100.0, 4),
            "expected_value_cents_total": total_expected_value_cents,
            "expected_minus_realized_cents": round(total_expected_value_cents - total_pnl_cents, 3),
            "avg_pnl_per_trade_cents": round(total_pnl_cents / trade_count, 3) if trade_count else None,
            "avg_edge_cents": round(sum(t.edge_cents for t in trades) / trade_count, 3) if trade_count else None,
            "avg_expected_value_cents": round(sum(t.expected_value_cents for t in trades) / trade_count, 3) if trade_count else None,
            "base_max_trades_per_ticker": int(max_trades_per_ticker),
            "base_min_reentry_minutes": int(min_reentry_minutes),
            "adaptive_lookback_days": int(adaptive_lookback_days),
        }

        return summary, trades
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Internal replay backtest runner for the remaining-only orchestrator path. "
            "For canonical model-comparison/reporting flows, use analyzer.canonical_replay_bundle."
        )
    )
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--date", action="append", dest="dates", help="Specific target_date(s), repeatable")
    parser.add_argument("--start-date", help="Inclusive lower date bound (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Inclusive upper date bound (YYYY-MM-DD)")
    parser.add_argument("--step-minutes", type=int, default=30, help="Replay cadence from snapshot timeline")
    parser.add_argument("--max-cycles-per-day", type=int, default=None)
    parser.add_argument("--max-trades-per-ticker", type=int, default=3)
    parser.add_argument("--min-reentry-minutes", type=int, default=90)
    parser.add_argument("--adaptive-lookback-days", type=int, default=2)
    parser.add_argument("--trades-out", help="Optional NDJSON output for simulated trades")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    try:
        dates = _settled_dates(conn, args.station, args.start_date, args.end_date, args.dates)
    finally:
        conn.close()

    summary, trades = run_replay(
        db_path=args.db,
        station=args.station,
        dates=dates,
        step_minutes=max(1, int(args.step_minutes)),
        max_cycles_per_day=args.max_cycles_per_day,
        max_trades_per_ticker=max(1, int(args.max_trades_per_ticker)),
        min_reentry_minutes=max(0, int(args.min_reentry_minutes)),
        adaptive_lookback_days=max(1, int(args.adaptive_lookback_days)),
    )

    if args.trades_out:
        out = Path(args.trades_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for trade in trades:
                fh.write(json.dumps(asdict(trade), sort_keys=True) + "\n")

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
