"""Prompt builder — assembles the LLM prompt from template + daily data.

Reads the instruction template from prompts/daily_review.md, inserts
current regime/signal family definitions, renders the DailyDataPackage,
and inserts recent review context plus statistical calibration context.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

def _build_live_regime_definitions() -> str:
    """Generate regime definitions from the live catalog."""
    try:
        from engine.regime_catalog import REGIME_PARAMS, RegimeType
        lines = []
        for rt in RegimeType:
            params = REGIME_PARAMS.get(rt, {})
            lines.append(f"### {rt.value}")
            lines.append(f"- sigma_multiplier: {params.get('sigma_multiplier', 1.0)}")
            lines.append(f"- mu_bias_high_f: {params.get('mu_bias_high_f', 0.0)}")
            lines.append(f"- mu_bias_low_f: {params.get('mu_bias_low_f', 0.0)}")
            lines.append(f"- min_confidence_for_sizing: {params.get('min_confidence_for_sizing', 0.3)}")
            lines.append("")

        # Also include promoted regimes from JSON
        promoted_path = Path("analysis_data/promoted_regimes.json")
        if promoted_path.exists():
            import json as _json
            promoted = _json.loads(promoted_path.read_text())
            if promoted:
                lines.append("### Promoted Regimes (from HMM/DS3M discovery)")
                for r in promoted:
                    lines.append(f"- **{r.get('name', '?')}**: sigma={r.get('sigma_multiplier', '?')}, "
                                 f"mu_high={r.get('mu_bias_high_f', 0)}, mu_low={r.get('mu_bias_low_f', 0)}")
                lines.append("")
        return "\n".join(lines)
    except ImportError:
        return "_Regime catalog not available._"


SIGNAL_FAMILY_DEFINITIONS = """
Signal families are now defined by the regime catalog + BOCPD changepoint detection.
The old signal_families.py pipeline has been removed. The current system uses:

- **Regime catalog** (regime_catalog.py): 6 core regimes with sigma/mu conditioning
- **BOCPD changepoint detection**: 3-layer changepoint system (thresholds + CUSUM + BOCPD)
- **LETKF spatial assimilation**: 14 SE Florida stations, sigma blending
- **DS3M particle filter**: auto-discovers new regimes from observation likelihood gaps

You are NOT limited to existing regimes. Propose new regimes freely in snake_case.
"""


def build_prompt(
    data_text: str,
    target_date: str,
    recent_context: str = "",
    *,
    template_path: Path | None = None,
    calibration_context: str = "",
    # Additional data section kwargs — each maps to a {{TOKEN}} in the template
    hdp_hmm_results: str = "",
    promoted_regimes: str = "",
    letkf_diagnostics: str = "",
    letkf_sigma_max_weight: str = "",
    letkf_min_updates: str = "",
    emos_state: str = "",
    boa_state: str = "",
    cusum_proposals: str = "",
    platt_state: str = "",
    market_density: str = "",
    paper_trade_summary: str = "",
    ds3m_paper_trading_comparison: str = "",
    ds3m_regime_discoveries: str = "",
    collection_health: str = "",
    n_settlements: str = "",
    ds3m_n_cycles: str = "",
    letkf_n_updates: str = "",
    ds3m_regimes_discovered: str = "",
    ds3m_unnamed_regimes: str = "",
    ds3m_vs_prod_crps: str = "",
    months_to_ds3m_production: str = "",
    exit_tuner_state: str = "",
) -> str:
    """Build the complete LLM prompt from template + data."""
    if template_path is None:
        template_path = Path(__file__).parent / "prompts" / "daily_review.md"

    if template_path.exists():
        template = template_path.read_text()
    else:
        log.warning("Prompt template not found at %s, using inline fallback", template_path)
        template = _FALLBACK_TEMPLATE

    prompt = template

    # Core tokens
    prompt = prompt.replace("{{REGIME_DEFINITIONS}}", _build_live_regime_definitions())
    prompt = prompt.replace("{{SIGNAL_FAMILY_DEFINITIONS}}", SIGNAL_FAMILY_DEFINITIONS.strip())
    prompt = prompt.replace("{{PREVIOUS_DAYS_CONTEXT}}", recent_context or "_No previous reviews available._")
    prompt = prompt.replace("{{DAILY_DATA}}", data_text)
    prompt = prompt.replace("{{TARGET_DATE}}", target_date)
    prompt = prompt.replace(
        "{{CALIBRATION_CONTEXT}}",
        calibration_context or "_No calibration data available yet._",
    )

    # Data section tokens — use provided text or a clear unavailable message
    _na = "_Data not available._"
    section_tokens = {
        "{{HDP_HMM_RESULTS}}": hdp_hmm_results,
        "{{PROMOTED_REGIMES}}": promoted_regimes,
        "{{LETKF_DIAGNOSTICS}}": letkf_diagnostics,
        "{{LETKF_SIGMA_MAX_WEIGHT}}": letkf_sigma_max_weight or "0.4",
        "{{LETKF_MIN_UPDATES}}": letkf_min_updates,
        "{{EMOS_STATE}}": emos_state,
        "{{BOA_STATE}}": boa_state,
        "{{CUSUM_PROPOSALS}}": cusum_proposals,
        "{{PLATT_STATE}}": platt_state,
        "{{MARKET_DENSITY}}": market_density,
        "{{PAPER_TRADE_SUMMARY}}": paper_trade_summary,
        "{{DS3M_PAPER_TRADING_COMPARISON}}": ds3m_paper_trading_comparison,
        "{{DS3M_REGIME_DISCOVERIES}}": ds3m_regime_discoveries,
        "{{COLLECTION_HEALTH}}": collection_health,
        "{{N_SETTLEMENTS}}": n_settlements,
        "{{DS3M_N_CYCLES}}": ds3m_n_cycles,
        "{{LETKF_N_UPDATES}}": letkf_n_updates,
        "{{DS3M_REGIMES_DISCOVERED}}": ds3m_regimes_discovered,
        "{{DS3M_UNNAMED_REGIMES}}": ds3m_unnamed_regimes,
        "{{DS3M_VS_PROD_CRPS}}": ds3m_vs_prod_crps,
        "{{MONTHS_TO_DS3M_PRODUCTION}}": months_to_ds3m_production,
        "{{EXIT_TUNER_STATE}}": exit_tuner_state,
        "{{ENTRY_TUNER_STATE}}": build_entry_tuner_state_text(),
    }
    for token, value in section_tokens.items():
        prompt = prompt.replace(token, value or _na)

    # Safety net: replace any remaining {{...}} tokens
    prompt = _replace_all_tokens(prompt)

    return prompt


def _replace_all_tokens(prompt: str) -> str:
    """Safety net — replace any remaining {{...}} patterns with a clear message."""
    def _sub(m: re.Match) -> str:
        token_name = m.group(1)
        log.warning("Unreplaced template token found: {{%s}}", token_name)
        return "_Data not available._"
    return re.sub(r"\{\{([A-Z_]+)\}\}", _sub, prompt)


# ---------------------------------------------------------------------------
# Data section builders — query DB/JSON and return formatted markdown
# ---------------------------------------------------------------------------

def build_ds3m_comparison_text(db_path: str | Path, station: str = "KMIA") -> str:
    """Query ds3m_comparison and ds3m_paper_trades tables for recent data."""
    db_path = Path(db_path)
    if not db_path.exists():
        return "_Data not available._"
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT target_date, ticker,
                      production_probability, ds3m_probability,
                      production_edge, ds3m_edge,
                      production_crps, ds3m_crps, actual_outcome
               FROM ds3m_comparison
               WHERE station = ?
               ORDER BY target_date DESC, ticker
               LIMIT 50""",
            (station,),
        ).fetchall()
        conn.close()
    except Exception as e:
        log.warning("Failed to query ds3m_comparison: %s", e)
        return "_Data not available._"

    if not rows:
        return "_No DS3M comparison data available yet._"

    lines = ["| Date | Ticker | Prod Prob | DS3M Prob | Prod Edge | DS3M Edge | Prod CRPS | DS3M CRPS | Outcome |",
             "|------|--------|-----------|-----------|-----------|-----------|-----------|-----------|---------|"]
    for r in rows:
        lines.append(
            f"| {r['target_date']} | {r['ticker']} | {_fmt(r['production_probability'])} "
            f"| {_fmt(r['ds3m_probability'])} | {_fmt(r['production_edge'])} "
            f"| {_fmt(r['ds3m_edge'])} | {_fmt(r['production_crps'])} "
            f"| {_fmt(r['ds3m_crps'])} | {_fmt(r['actual_outcome'])} |"
        )
    return "\n".join(lines)


def build_paper_trade_summary_text(db_path: str | Path, station: str = "KMIA") -> str:
    """Query paper_trades table for recent summary."""
    db_path = Path(db_path)
    if not db_path.exists():
        return "_Data not available._"
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT target_date, ticker, side, status,
                      entry_price_cents, exit_price_cents, realized_pnl_cents,
                      estimated_probability, expected_edge_cents
               FROM paper_trades
               WHERE station = ?
               ORDER BY target_date DESC, id DESC
               LIMIT 30""",
            (station,),
        ).fetchall()
        conn.close()
    except Exception as e:
        log.warning("Failed to query paper_trades: %s", e)
        return "_Data not available._"

    if not rows:
        return "_No paper trades recorded yet._"

    lines = ["| Date | Ticker | Side | Status | Entry | Exit | PnL | Est Prob | Edge |",
             "|------|--------|------|--------|-------|------|-----|----------|------|"]
    for r in rows:
        lines.append(
            f"| {r['target_date']} | {r['ticker']} | {r['side']} | {r['status']} "
            f"| {_fmt(r['entry_price_cents'])} | {_fmt(r['exit_price_cents'])} "
            f"| {_fmt(r['realized_pnl_cents'])} | {_fmt(r['estimated_probability'])} "
            f"| {_fmt(r['expected_edge_cents'])} |"
        )
    return "\n".join(lines)


def build_emos_state_text(base_dir: Path = Path("analysis_data")) -> str:
    """Load analysis_data/emos_state.json."""
    data = _load_json_dict(base_dir / "emos_state.json")
    if not data:
        return "_EMOS state not available._"
    lines = ["EMOS calibration state:"]
    for k, v in data.items():
        lines.append(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return "\n".join(lines)


def build_boa_state_text(base_dir: Path = Path("analysis_data")) -> str:
    """Load analysis_data/boa_state.json."""
    data = _load_json_dict(base_dir / "boa_state.json")
    if not data:
        return "_BOA state not available._"
    lines = ["BOA source weights:"]
    for k, v in data.items():
        lines.append(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return "\n".join(lines)


def build_platt_state_text(base_dir: Path = Path("analysis_data")) -> str:
    """Load analysis_data/platt_state.json."""
    data = _load_json_dict(base_dir / "platt_state.json")
    if not data:
        return "_Platt calibration state not available._"
    lines = ["Platt calibration state:"]
    for k, v in data.items():
        lines.append(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return "\n".join(lines)


def build_ds3m_regime_discoveries_text(base_dir: Path = Path("analysis_data")) -> str:
    """Load analysis_data/ds3m_regime_gaps.json."""
    data = _load_json_dict(base_dir / "ds3m_regime_gaps.json")
    if not data:
        return "_No DS3M regime discovery data available._"
    lines = ["DS3M regime discovery gaps:"]
    for k, v in data.items():
        lines.append(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}")
    return "\n".join(lines)


def build_collection_health_text(db_path: str | Path) -> str:
    """Query collection_runs table for recent health status."""
    db_path = Path(db_path)
    if not db_path.exists():
        return "_Data not available._"
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT collector, status, started_at, records_collected, error_text
               FROM collection_runs
               ORDER BY started_at DESC
               LIMIT 30""",
        ).fetchall()
        conn.close()
    except Exception as e:
        log.warning("Failed to query collection_runs: %s", e)
        return "_Data not available._"

    if not rows:
        return "_No collection run data available._"

    # Summarize by collector (most recent run per collector)
    collectors: dict[str, dict] = {}
    for r in rows:
        name = r["collector"]
        if name not in collectors:
            collectors[name] = {
                "last_status": r["status"],
                "last_run": r["started_at"],
                "last_records": r["records_collected"],
                "last_error": r["error_text"],
            }

    lines = ["| Collector | Last Status | Last Run | Records | Error |",
             "|-----------|-------------|----------|---------|-------|"]
    for name, info in sorted(collectors.items()):
        err = (info["last_error"] or "")[:60]
        lines.append(
            f"| {name} | {info['last_status']} | {info['last_run']} "
            f"| {info['last_records']} | {err} |"
        )
    return "\n".join(lines)


def build_ds3m_tracker_text(db_path: str | Path, station: str = "KMIA") -> str:
    """Query ds3m_estimates for cycle count, settlements, etc."""
    db_path = Path(db_path)
    if not db_path.exists():
        return "_Data not available._"
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        n_cycles = conn.execute(
            "SELECT COUNT(DISTINCT timestamp_utc) FROM ds3m_estimates WHERE station = ?",
            (station,),
        ).fetchone()[0]
        n_dates = conn.execute(
            "SELECT COUNT(DISTINCT target_date) FROM ds3m_estimates WHERE station = ?",
            (station,),
        ).fetchone()[0]
        latest_regime = conn.execute(
            "SELECT regime_posterior, ess FROM ds3m_estimates WHERE station = ? ORDER BY timestamp_utc DESC LIMIT 1",
            (station,),
        ).fetchone()
        conn.close()
    except Exception as e:
        log.warning("Failed to query ds3m_estimates: %s", e)
        return "_Data not available._"

    lines = [
        f"- DS3M shadow cycles completed: {n_cycles}",
        f"- DS3M target dates covered: {n_dates}",
    ]
    if latest_regime:
        lines.append(f"- Latest ESS: {latest_regime[1]}")
        if latest_regime[0]:
            try:
                posterior = json.loads(latest_regime[0]) if isinstance(latest_regime[0], str) else latest_regime[0]
                n_regimes = len(posterior) if isinstance(posterior, dict) else "?"
                lines.append(f"- Regime posterior entries: {n_regimes}")
            except (json.JSONDecodeError, TypeError):
                pass

    return "\n".join(lines)


def build_hdp_hmm_results_text(db_path: str | Path, station: str = "KMIA", target_date: str | None = None) -> str:
    """Summarize latest HDP shadow output from regime_labels_hdp_test."""
    db_path = Path(db_path)
    if not db_path.exists():
        return "_No HDP-HMM shadow results available yet._"

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        if target_date:
            row = conn.execute(
                """SELECT target_date, n_regimes, regime_sequence, phase_summary, transition_matrix
                   FROM regime_labels_hdp_test
                   WHERE station = ? AND target_date = ?
                   ORDER BY id DESC LIMIT 1""",
                (station, target_date),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT target_date, n_regimes, regime_sequence, phase_summary, transition_matrix
                   FROM regime_labels_hdp_test
                   WHERE station = ?
                   ORDER BY target_date DESC, id DESC LIMIT 1""",
                (station,),
            ).fetchone()
        conn.close()
    except Exception as e:
        log.warning("Failed to query regime_labels_hdp_test: %s", e)
        return "_No HDP-HMM shadow results available yet._"

    if row is None:
        return "_No HDP-HMM shadow results available yet._"

    seq_preview = ""
    if row["regime_sequence"]:
        try:
            seq = json.loads(row["regime_sequence"]) if isinstance(row["regime_sequence"], str) else row["regime_sequence"]
            if isinstance(seq, list) and seq:
                seq_preview = ", ".join(str(x) for x in seq[:18])
                if len(seq) > 18:
                    seq_preview += ", ..."
        except (json.JSONDecodeError, TypeError):
            pass

    lines = [
        f"- HDP shadow date: {row['target_date']}",
        f"- Regimes discovered: {row['n_regimes']}",
    ]
    if seq_preview:
        lines.append(f"- Regime sequence sample: [{seq_preview}]")
    if row["phase_summary"]:
        text = str(row["phase_summary"])
        lines.append(f"- Phase summary: {text[:360]}{'...' if len(text) > 360 else ''}")
    return "\n".join(lines)


def build_market_density_text(db_path: str | Path, station: str = "KMIA", target_date: str | None = None) -> str:
    """Summarize snapshot/quote density for recent market data health."""
    db_path = Path(db_path)
    if not db_path.exists():
        return "_Data not available._"
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        if target_date:
            rows = conn.execute(
                """SELECT forecast_date,
                          COUNT(*) AS snapshots,
                          COUNT(DISTINCT ticker) AS tickers,
                          AVG(CASE WHEN best_yes_bid_cents IS NOT NULL AND best_yes_ask_cents IS NOT NULL
                                   THEN best_yes_ask_cents - best_yes_bid_cents END) AS avg_yes_spread,
                          AVG(CASE WHEN best_no_bid_cents IS NOT NULL AND best_no_ask_cents IS NOT NULL
                                   THEN best_no_ask_cents - best_no_bid_cents END) AS avg_no_spread
                   FROM market_snapshots
                   WHERE forecast_date = ?
                   GROUP BY forecast_date""",
                (target_date,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT forecast_date,
                          COUNT(*) AS snapshots,
                          COUNT(DISTINCT ticker) AS tickers,
                          AVG(CASE WHEN best_yes_bid_cents IS NOT NULL AND best_yes_ask_cents IS NOT NULL
                                   THEN best_yes_ask_cents - best_yes_bid_cents END) AS avg_yes_spread,
                          AVG(CASE WHEN best_no_bid_cents IS NOT NULL AND best_no_ask_cents IS NOT NULL
                                   THEN best_no_ask_cents - best_no_bid_cents END) AS avg_no_spread
                   FROM market_snapshots
                   GROUP BY forecast_date
                   ORDER BY forecast_date DESC
                   LIMIT 7""",
            ).fetchall()
        conn.close()
    except Exception as e:
        log.warning("Failed to query market density: %s", e)
        return "_Data not available._"

    if not rows:
        return "_No market snapshot density data available._"

    lines = [
        "| Date | Snapshots | Tickers | Avg YES spread | Avg NO spread |",
        "|------|-----------|---------|----------------|---------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['forecast_date']} | {r['snapshots']} | {r['tickers']} "
            f"| {_fmt(r['avg_yes_spread'])} | {_fmt(r['avg_no_spread'])} |"
        )
    return "\n".join(lines)


def build_letkf_diagnostics_text(base_dir: Path = Path("analysis_data")) -> str:
    """Summarize LETKF tune state for prompt diagnostics."""
    data = _load_json_dict(base_dir / "letkf_tune_state.json")
    if not data:
        return "_LETKF diagnostics not available._"

    n_updates = data.get("n_settlements")
    if n_updates is None:
        hist = data.get("history")
        n_updates = len(hist) if isinstance(hist, list) else 0

    lines = [
        f"- n_updates: {n_updates}",
        f"- sigma_max_weight: {_fmt(data.get('sigma_max_weight'))}",
        f"- min_updates: {_fmt(data.get('min_updates'))}",
    ]
    if data.get("last_update_utc"):
        lines.append(f"- last_update_utc: {data['last_update_utc']}")
    return "\n".join(lines)


def build_letkf_tracker_metrics(base_dir: Path = Path("analysis_data")) -> dict[str, str]:
    """Extract numeric LETKF tracker values used by template tokens."""
    data = _load_json_dict(base_dir / "letkf_tune_state.json")
    if not data:
        return {"sigma_max_weight": "0.4", "min_updates": "_Data not available._", "n_updates": "_Data not available._"}

    n_updates = data.get("n_settlements")
    if n_updates is None:
        hist = data.get("history")
        n_updates = len(hist) if isinstance(hist, list) else "_Data not available._"

    return {
        "sigma_max_weight": str(data.get("sigma_max_weight", 0.4)),
        "min_updates": str(data.get("min_updates", "_Data not available._")),
        "n_updates": str(n_updates),
    }


def build_entry_tuner_state_text(base_dir: Path = Path("analysis_data")) -> str:
    """Summarize production + DS3M entry tuner states."""
    def _state_summary(path: Path, label: str) -> list[str]:
        data = _load_json_dict(path)
        if not data:
            return [f"- {label}: _not available_"]
        lines = [
            f"- {label}: n_evaluations={data.get('n_evaluations', 0)}, last={data.get('last_evaluation_utc', 'n/a')}",
        ]
        params = data.get("current_params") or {}
        if params:
            lines.append(
                "  - params: "
                f"edge={_fmt(params.get('min_edge_cents'))}, "
                f"ev={_fmt(params.get('min_ev_cents'))}, "
                f"max_price={_fmt(params.get('max_entry_price_cents'))}, "
                f"min_price={_fmt(params.get('min_entry_price_cents'))}"
            )
        return lines

    lines: list[str] = []
    lines.extend(_state_summary(base_dir / "entry_tuner_state.json", "production"))
    lines.extend(_state_summary(base_dir / "ds3m_entry_tuner_state.json", "ds3m"))
    return "\n".join(lines)


def build_exit_tuner_state_text(base_dir: Path = Path("analysis_data")) -> str:
    """Summarize production + DS3M exit tuner states."""

    def _state_summary(path: Path, label: str) -> list[str]:
        data = _load_json_dict(path)
        if not data:
            return [f"- {label}: _not available_"]
        lines = [
            f"- {label}: n_evaluations={data.get('n_evaluations', 0)}, last={data.get('last_evaluation_utc', 'n/a')}",
        ]
        params = data.get("current_params") or {}
        if params:
            lines.append(
                "  - params: "
                f"risk_frac={_fmt(params.get('profit_take_risk_fraction'))}, "
                f"time_decay={_fmt(params.get('profit_take_time_decay'))}, "
                f"buffer={_fmt(params.get('deterioration_buffer_cents'))}, "
                f"min_edge={_fmt(params.get('min_edge_cents'))}, "
                f"min_ev={_fmt(params.get('min_ev_cents'))}"
            )
        history = data.get("evaluation_history") or []
        if history:
            last = history[-1]
            lines.append(
                "  - latest_eval: "
                f"n_trades={last.get('n_trades')}, "
                f"current_pnl={_fmt(last.get('current_pnl'))}, "
                f"best_sharpe_pnl={_fmt(last.get('best_sharpe_pnl'))}, "
                f"best_sharpe={_fmt(last.get('best_sharpe_ratio'))}"
            )
        return lines

    lines: list[str] = []
    lines.extend(_state_summary(base_dir / "exit_tuner_state.json", "production"))
    lines.extend(_state_summary(base_dir / "ds3m_exit_tuner_state.json", "ds3m"))
    return "\n".join(lines)


def build_ds3m_progress_metrics(
    db_path: str | Path,
    station: str = "KMIA",
    base_dir: Path = Path("analysis_data"),
) -> dict[str, str]:
    """Compute DS3M progress tracker token values."""
    metrics = {
        "n_settlements": "_Data not available._",
        "ds3m_n_cycles": "_Data not available._",
        "ds3m_regimes_discovered": "_Data not available._",
        "ds3m_unnamed_regimes": "_Data not available._",
        "ds3m_vs_prod_crps": "_Data not available._",
        "months_to_ds3m_production": "TBD (requires explicit promotion criteria)",
    }

    db_path = Path(db_path)
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=10)
            n_settlements = conn.execute(
                "SELECT COUNT(DISTINCT forecast_date) FROM market_settlements WHERE station = ?",
                (station,),
            ).fetchone()[0]
            n_cycles = conn.execute(
                "SELECT COUNT(DISTINCT timestamp_utc) FROM ds3m_estimates WHERE station = ?",
                (station,),
            ).fetchone()[0]
            crps = conn.execute(
                """SELECT AVG(ds3m_crps) AS ds3m_crps_avg, AVG(production_crps) AS prod_crps_avg
                   FROM ds3m_comparison
                   WHERE station = ? AND ds3m_crps IS NOT NULL AND production_crps IS NOT NULL""",
                (station,),
            ).fetchone()
            conn.close()

            metrics["n_settlements"] = str(n_settlements)
            metrics["ds3m_n_cycles"] = str(n_cycles)
            if crps and crps[0] is not None and crps[1] is not None:
                delta = float(crps[0]) - float(crps[1])
                metrics["ds3m_vs_prod_crps"] = (
                    f"ds3m={crps[0]:.4f}, prod={crps[1]:.4f}, delta(ds3m-prod)={delta:+.4f}"
                )
        except Exception as e:
            log.warning("Failed to compute DS3M progress metrics: %s", e)

    ds3m_state = _load_json_dict(base_dir / "ds3m_state.json")
    if ds3m_state:
        dyn = ds3m_state.get("dynamics_dict") or {}
        names = dyn.get("regime_names") or []
        if names:
            metrics["ds3m_regimes_discovered"] = str(len(names))
            unnamed = [n for n in names if isinstance(n, str) and n.startswith("regime_")]
            metrics["ds3m_unnamed_regimes"] = ", ".join(unnamed) if unnamed else "none"

    return metrics


def build_calibration_context(
    live_cusum_config_path: Path = Path("analysis_data/cusum_config.json"),
    proposed_cusum_config_path: Path = Path("analysis_data/cusum_config.proposed.json"),
    live_skf_config_path: Path = Path("analysis_data/skf_config.json"),
    proposed_skf_config_path: Path = Path("analysis_data/skf_config.proposed.json"),
    calibration_log_path: Path = Path("analysis_data/calibration_log.jsonl"),
    letkf_tune_state_path: Path = Path("analysis_data/letkf_tune_state.json"),
) -> str:
    """Build statistical calibration context for the AI prompt.

    The key point is that the deterministic statistical step has already run.
    The AI should audit the proposal, not blindly overwrite it.
    """
    lines = [
        "The deterministic statistical correction step already ran before this review.",
        "Your job is to audit whether the proposal should be approved, dampened, held, or rejected.",
        "Treat single-day overreaction risk seriously.",
        "",
    ]

    live_cusum = _load_json_dict(live_cusum_config_path)
    proposed_cusum = _load_json_dict(proposed_cusum_config_path)
    live_skf = _load_json_dict(live_skf_config_path)
    proposed_skf = _load_json_dict(proposed_skf_config_path)

    if live_cusum or proposed_cusum:
        lines.append("### CUSUM Statistical Proposal")
        lines.append("| Channel | Current h | Proposed h | Delta | k |")
        lines.append("|---------|-----------|------------|-------|---|")
        current_h = (live_cusum or {}).get("h", {})
        proposed_h = (proposed_cusum or live_cusum or {}).get("h", {})
        current_k = (live_cusum or proposed_cusum or {}).get("k", {})
        for channel in sorted(set(current_h) | set(proposed_h) | set(current_k)):
            cur_h = current_h.get(channel)
            prop_h = proposed_h.get(channel, cur_h)
            delta = _format_delta(cur_h, prop_h)
            lines.append(
                f"| {channel} | {_fmt(cur_h)} | {_fmt(prop_h)} | {delta} | {_fmt(current_k.get(channel))} |"
            )
        lines.append("")
        lines.append(
            "Audit question: was the CUSUM sensitivity move directionally correct, overreactionary, or underwhelming?"
        )
        lines.append("")

    if live_skf or proposed_skf:
        lines.append("### SKF Statistical Proposal (correction terms only)")
        lines.append(
            "| Regime | μ_high current | μ_high proposed | μ_low current | μ_low proposed | σ_high current | σ_high proposed | σ_low current | σ_low proposed |"
        )
        lines.append(
            "|--------|----------------|-----------------|---------------|----------------|----------------|-----------------|---------------|----------------|"
        )
        live_regimes = _regime_map(live_skf)
        proposed_regimes = _regime_map(proposed_skf or live_skf)
        for name in sorted(set(live_regimes) | set(proposed_regimes)):
            cur = live_regimes.get(name, {})
            prop = proposed_regimes.get(name, cur)
            lines.append(
                "| {name} | {mh_cur} | {mh_prop} | {ml_cur} | {ml_prop} | {sh_cur} | {sh_prop} | {sl_cur} | {sl_prop} |".format(
                    name=name,
                    mh_cur=_fmt(cur.get("mu_shift_high")),
                    mh_prop=_fmt(prop.get("mu_shift_high")),
                    ml_cur=_fmt(cur.get("mu_shift_low")),
                    ml_prop=_fmt(prop.get("mu_shift_low")),
                    sh_cur=_fmt(cur.get("sigma_scale_high")),
                    sh_prop=_fmt(prop.get("sigma_scale_high")),
                    sl_cur=_fmt(cur.get("sigma_scale_low")),
                    sl_prop=_fmt(prop.get("sigma_scale_low")),
                )
            )
        lines.append("")
        lines.append(
            "Audit question: were the proposed mu/sigma correction moves reasonable for the dominant regime, or did the stats overfit one day?"
        )
        lines.append("")

    if calibration_log_path.exists():
        recent = []
        with open(calibration_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    recent.append(json.loads(line))
        recent = recent[-3:]
        if recent:
            lines.append("### Recent Calibration History (last 3 entries)")
            for entry in recent:
                date = entry.get("target_date", "?")
                skf_info = entry.get("skf", {})
                cusum_info = entry.get("cusum", {})
                lines.append(
                    f"- **{date}**: dominant_regime={skf_info.get('dominant_regime', '?')}, "
                    f"high_err={skf_info.get('error_high', '?')}, "
                    f"low_err={skf_info.get('error_low', '?')}, "
                    f"CUSUM updates={cusum_info.get('channels_updated', 0)}"
                )
            lines.append("")

    # LETKF spatial assimilation diagnostics
    letkf_tune = _load_json_dict(letkf_tune_state_path)
    if letkf_tune:
        lines.append("### LETKF Spatial Assimilation Diagnostics")
        lines.append(f"Settlements processed: {letkf_tune.get('n_settlements', 0)}")
        lines.append("")

        # Per-station table: spread, R estimates, innovation stats
        lines.append("| Station | Spread | R (current) | R (initial) | R drift | High inn mean | High inn std | Low inn mean | Low inn std | Flagged |")
        lines.append("|---------|--------|-------------|-------------|---------|--------------|-------------|-------------|------------|---------|")

        inn_high = letkf_tune.get("innovation_history_high", {})
        inn_low = letkf_tune.get("innovation_history_low", {})
        initial_r = letkf_tune.get("initial_r_estimates", {})

        # We don't have live LETKFState in the prompt builder, so use tune_state
        # fields only. Analysis spread comes from the last inference cycle's notes.
        for stn_code in sorted(set(list(inn_high.keys()) + list(inn_low.keys()) + list(initial_r.keys()))):
            init_r = initial_r.get(stn_code)
            # Compute innovation stats from history
            h_vals = inn_high.get(stn_code, [])
            l_vals = inn_low.get(stn_code, [])

            h_mean = f"{sum(h_vals) / len(h_vals):+.2f}" if h_vals else "-"
            h_std = f"{_std(h_vals):.2f}" if len(h_vals) >= 2 else "-"
            l_mean = f"{sum(l_vals) / len(l_vals):+.2f}" if l_vals else "-"
            l_std = f"{_std(l_vals):.2f}" if len(l_vals) >= 2 else "-"

            # R drift detection (can't compute current_r without live state,
            # but we can flag if innovations suggest drift)
            r_drift_flag = ""
            if init_r is not None and h_vals:
                # Approximate: if mean abs innovation >> initial R, flag it
                mean_abs_inn = sum(abs(v) for v in h_vals) / len(h_vals)
                if mean_abs_inn > init_r + _LETKF_R_DRIFT_THRESHOLD:
                    r_drift_flag = "DRIFT"

            lines.append(
                f"| {stn_code} | - | - | {_fmt(init_r)} | - | {h_mean} | {h_std} | {l_mean} | {l_std} | {r_drift_flag} |"
            )

        lines.append("")
        lines.append(
            "Audit question: are LETKF innovations showing systematic bias that needs R adjustment, "
            "or is the spatial assimilation tracking well?"
        )
        lines.append("")

    return "\n".join(lines).strip() if lines else ""


# Threshold for flagging R drift in prompt builder (matches letkf.py _R_DRIFT_THRESHOLD)
_LETKF_R_DRIFT_THRESHOLD = 0.5


def _std(vals: list) -> float:
    """Compute sample std for a list of floats."""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def build_recent_context(recent_labels: list[dict]) -> str:
    """Format recent regime labels into context text for the prompt."""
    if not recent_labels:
        return "_No previous reviews available._"

    lines = []
    for label in recent_labels:
        date = label.get("target_date", "?")
        regimes = label.get("regimes_active", "[]")
        path = label.get("path_class", "?")
        source = label.get("review_source", "?")

        if isinstance(regimes, str):
            try:
                regimes = json.loads(regimes)
            except (json.JSONDecodeError, TypeError):
                regimes = [regimes]

        regime_str = ", ".join(regimes) if isinstance(regimes, list) else str(regimes)

        phases = label.get("phase_summary")
        phase_str = ""
        if phases:
            if isinstance(phases, str):
                try:
                    phases = json.loads(phases)
                except (json.JSONDecodeError, TypeError):
                    phases = None
            if isinstance(phases, list):
                phase_str = "\n".join(
                    f"    - {p.get('start_hour_lst', '?')}:00–{p.get('end_hour_lst', '?')}:00 LST: "
                    f"{p.get('regime', '?')} — {p.get('description', '')}"
                    for p in phases
                )

        lines.append(f"### {date} (source: {source})")
        lines.append(f"- **Regimes:** {regime_str}")
        lines.append(f"- **Path:** {path}")
        if phase_str:
            lines.append(f"- **Phases:**\n{phase_str}")
        lines.append("")

    return "\n".join(lines)


def save_prompt(prompt: str, target_date: str, output_dir: Path) -> Path:
    """Write the assembled prompt to disk for external consumption."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"review_prompt_{target_date}.md"
    path.write_text(prompt)
    log.info("Saved review prompt to %s (%d chars)", path, len(prompt))
    return path


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("Could not parse JSON config at %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _regime_map(raw: dict) -> dict[str, dict]:
    regimes = raw.get("regimes", raw) if isinstance(raw, dict) else {}
    if isinstance(regimes, dict):
        regime_list = list(regimes.values())
    elif isinstance(regimes, list):
        regime_list = regimes
    else:
        regime_list = []
    result: dict[str, dict] = {}
    for regime in regime_list:
        if isinstance(regime, dict) and regime.get("name"):
            result[str(regime["name"])] = regime
    return result


def _fmt(value) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_delta(current, proposed) -> str:
    if current is None or proposed is None:
        return "?"
    try:
        delta = float(proposed) - float(current)
    except (TypeError, ValueError):
        return "?"
    return f"{delta:+.3f}"


_FALLBACK_TEMPLATE = """# KMIA Post-Settlement Daily Review

Analyze the completed climate day data below and produce a structured review.

{{REGIME_DEFINITIONS}}

{{SIGNAL_FAMILY_DEFINITIONS}}

## Statistical Calibration Context

{{CALIBRATION_CONTEXT}}

## Previous Days' Context

{{PREVIOUS_DAYS_CONTEXT}}

## Today's Data

{{DAILY_DATA}}

Produce output with YAML sections for: structured summary, phase breakdown,
signal labels, signal families, model performance, statistical audit,
threshold recommendations, and a narrative review.
"""
