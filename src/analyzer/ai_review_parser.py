"""AI review parser — parse structured LLM output and store results.

Parses the 7-section output format (YAML blocks + narrative markdown)
produced by the daily review LLM prompt. Stores structured data in the
regime_labels table, statistical audit history in analysis_data/, and
narrative in reviews/ai/.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ParsedReview:
    """Structured output from a parsed AI review."""
    station: str = "KMIA"
    target_date: str = ""
    cli_high_f: float | None = None
    cli_low_f: float | None = None
    regimes_active: list[str] = field(default_factory=list)
    path_class: str | None = None
    confidence_tags: list[str] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)
    signal_labels: list[dict] = field(default_factory=list)
    families_active: list[dict] = field(default_factory=list)
    model_performance: list[dict] = field(default_factory=list)
    statistical_audit: dict = field(default_factory=dict)
    threshold_recommendations: list[dict] = field(default_factory=list)
    narrative_markdown: str = ""
    # New fields from unified prompt
    regime_catalog_assessment: dict = field(default_factory=dict)
    letkf_recommendations: dict = field(default_factory=dict)
    paper_trading_comparison: dict = field(default_factory=dict)
    ds3m_comparison: dict = field(default_factory=dict)
    distribution_assessment: dict = field(default_factory=dict)
    priority_actions: list[dict] = field(default_factory=list)
    hold_items: list[dict] = field(default_factory=list)
    dismissed_items: list[dict] = field(default_factory=list)
    ds3m_progress: dict = field(default_factory=dict)
    data_health: dict = field(default_factory=dict)


def parse_review(raw_output: str) -> ParsedReview:
    """Parse the raw LLM output into a structured ParsedReview.

    Extracts YAML blocks from ```yaml fences and the narrative markdown
    from after the last YAML block. Lenient and backward-compatible — old
    6-block reviews and newer reviews with a statistical audit block are
    both accepted.
    """
    review = ParsedReview()

    # Extract all YAML blocks and classify by content (key-based, not positional)
    yaml_blocks = _extract_yaml_blocks(raw_output)

    for idx in range(len(yaml_blocks)):
        block = _safe_parse_yaml(yaml_blocks, idx)
        if not block:
            continue
        block_raw = block  # keep original for unwrapping
        block = _unwrap_single_key(block) if isinstance(block, dict) else block

        if not isinstance(block, dict):
            continue

        keys = set(block.keys())

        # Structured summary: contains station + cli_high_f or regimes_active
        if keys & {"station", "cli_high_f", "cli_low_f", "regimes_active"}:
            review.station = block.get("station", "KMIA")
            review.target_date = str(block.get("target_date") or block.get("date", ""))
            if not review.target_date and block.get("utc_window"):
                m = re.search(r"(\d{4}-\d{2}-\d{2})", str(block["utc_window"]))
                if m:
                    review.target_date = m.group(1)
            review.cli_high_f = block.get("cli_high_f") or block.get("cli_settlement_high")
            review.cli_low_f = block.get("cli_low_f") or block.get("cli_settlement_low")
            regimes = block.get("regimes_active")
            if regimes is None:
                regimes = []
                if block.get("primary_regime"):
                    regimes.append(block["primary_regime"])
            review.regimes_active = regimes or []
            review.path_class = block.get("path_class")
            review.confidence_tags = block.get("confidence_tags", [])
            continue

        # Phase breakdown
        if "phases" in block:
            review.phases = _find_list(block, "phases", "phase_breakdown")
            continue

        # Signal labels
        sigs = _find_list(block, "signals", "signal_labels")
        if sigs and any(isinstance(s, dict) and "time_lst" in s for s in sigs[:3]):
            review.signal_labels = sigs
            continue

        # Signal families
        fams = _find_list(block, "families", "signal_families_active")
        if fams and any(isinstance(f, dict) and "strength" in f for f in fams[:3]):
            review.families_active = fams
            continue

        # Model performance
        models = _find_list(block, "models", "model_performance")
        if models and any(isinstance(m, dict) and ("forecast_high_f" in m or "high_error_f" in m) for m in models[:3]):
            review.model_performance = models
            continue

        # Statistical audit (new nested format with legacy fields)
        if _looks_like_statistical_audit(block):
            audit = block.get("statistical_audit", block)
            review.statistical_audit = audit
            continue

        # Threshold recommendations
        recs = _find_list(block, "recommendations", "threshold_recommendations")
        if recs and any(isinstance(r, dict) and "parameter" in r for r in recs[:3]):
            review.threshold_recommendations = recs
            continue

        # Regime catalog assessment
        if "regime_catalog" in block or "production_assessment" in block:
            review.regime_catalog_assessment = block.get("regime_catalog", block)
            continue

        # LETKF recommendations
        if "letkf_recommendations" in block or "weight_cap" in block:
            review.letkf_recommendations = block.get("letkf_recommendations", block)
            continue

        # Paper trading comparison
        if "paper_trading_comparison" in block:
            review.paper_trading_comparison = block["paper_trading_comparison"]
            continue

        # DS3M comparison
        if "ds3m_comparison" in block:
            review.ds3m_comparison = block["ds3m_comparison"]
            continue

        # Distribution assessment
        if "distribution_assessment" in block or "mixture_model" in block:
            review.distribution_assessment = block.get("distribution_assessment", block)
            continue

        # Priority actions
        if "priority_actions" in block:
            review.priority_actions = _find_list(block, "priority_actions")
            continue

        # Hold items
        if "hold_items" in block:
            review.hold_items = _find_list(block, "hold_items")
            continue

        # Dismissed items
        if "dismissed_items" in block:
            review.dismissed_items = _find_list(block, "dismissed_items")
            continue

        # DS3M progress
        if "ds3m_progress" in block:
            review.ds3m_progress = block["ds3m_progress"]
            continue

        # Data health
        if "data_health" in block:
            review.data_health = block["data_health"]
            continue

    # Narrative — everything after the last YAML block
    review.narrative_markdown = _extract_narrative(raw_output)

    return review


def store_review(
    parsed: ParsedReview,
    db_path: str | Path,
    reviews_dir: Path | None = None,
    threshold_history_path: Path | None = None,
    statistical_audit_history_path: Path | None = None,
) -> None:
    """Store the parsed review in the DB and filesystem.

    - Upserts into regime_labels table
    - Writes narrative to reviews/ai/{target_date}.md
    - Appends threshold recommendations to threshold_history.jsonl
    - Appends statistical audit decisions to statistical_audit_history.jsonl
    """
    db_path = Path(db_path)

    if not parsed.target_date:
        raise ValueError("Parsed review missing target_date; refusing to write ambiguous '.md' output")

    # 1. Store in regime_labels
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO regime_labels
               (station, target_date, regimes_active, path_class,
                confidence_tags, phase_summary, model_performance,
                signal_labels, signal_families_active,
                threshold_recommendations, review_path, review_source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                parsed.station,
                parsed.target_date,
                json.dumps(parsed.regimes_active),
                parsed.path_class,
                json.dumps(parsed.confidence_tags),
                json.dumps(parsed.phases),
                json.dumps(_models_to_dict(parsed.model_performance)),
                json.dumps(parsed.signal_labels),
                json.dumps(parsed.families_active),
                json.dumps(parsed.threshold_recommendations),
                f"reviews/ai/{parsed.target_date}.md",
                "ai",
            ),
        )
        conn.commit()
        log.info("Stored regime label for %s %s", parsed.station, parsed.target_date)
    finally:
        conn.close()

    # 2. Write narrative
    if reviews_dir is None:
        reviews_dir = db_path.parent / "reviews" / "ai"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    review_path = reviews_dir / f"{parsed.target_date}.md"
    review_path.write_text(parsed.narrative_markdown)
    log.info("Wrote narrative review to %s", review_path)

    # 3. Append threshold recommendations
    if parsed.threshold_recommendations:
        if threshold_history_path is None:
            threshold_history_path = db_path.parent / "analysis_data" / "threshold_history.jsonl"
        threshold_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(threshold_history_path, "a") as f:
            entry = {
                "target_date": parsed.target_date,
                "station": parsed.station,
                "recommendations": parsed.threshold_recommendations,
            }
            f.write(json.dumps(entry) + "\n")
        log.info("Appended %d threshold recommendations", len(parsed.threshold_recommendations))

    # 4. Append statistical audit history
    if parsed.statistical_audit:
        if statistical_audit_history_path is None:
            statistical_audit_history_path = db_path.parent / "analysis_data" / "statistical_audit_history.jsonl"
        statistical_audit_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(statistical_audit_history_path, "a") as f:
            entry = {
                "target_date": parsed.target_date,
                "station": parsed.station,
                "statistical_audit": parsed.statistical_audit,
            }
            f.write(json.dumps(entry) + "\n")
        log.info("Appended statistical audit for %s", parsed.target_date)


def backfill_from_human_reviews(
    reviews_dir: Path,
    db_path: str | Path,
    station: str = "KMIA",
) -> int:
    """Parse existing human-written reviews and seed the regime_labels table.

    Extracts regime, path_class, and phase information from the structured
    markdown format used in reviews/2026-03-*.md files.
    """
    db_path = Path(db_path)
    review_files = sorted(reviews_dir.glob("2026-*.md"))
    count = 0

    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        for review_path in review_files:
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", review_path.name)
            if not date_match:
                continue
            target_date = date_match.group(1)
            text = review_path.read_text()

            # Extract regime info from the structured summary section
            regimes = []
            path_class = None
            phases = []

            # Primary regime
            primary = re.search(r"\*\*Primary regime:\*\*\s*`([^`]+)`", text)
            if primary:
                regimes.append(primary.group(1))
            # Background regime
            bg = re.search(r"\*\*Background regime:\*\*\s*`([^`]+)`", text)
            if bg:
                regimes.append(bg.group(1))
            # Path class
            pc = re.search(r"\*\*Path class:\*\*\s*`([^`]+)`", text)
            if pc:
                path_class = pc.group(1)

            # Extract phases from "### Phase N" headers
            phase_pattern = re.finditer(
                r"### Phase \d+\s*[—–-]\s*(.*?)\((\d+):?\d*\s*(?:AM|PM).*?[—–-]\s*(\d+):?\d*\s*(?:AM|PM).*?\)\s*\n(.*?)(?=### Phase|\n## |\Z)",
                text,
                re.DOTALL,
            )
            for m in phase_pattern:
                desc = m.group(1).strip().rstrip("(")
                phases.append({
                    "description": desc,
                    "regime": regimes[0] if regimes else "unknown",
                })

            if not regimes:
                # Try to extract from other formats
                regime_match = re.search(r"regime.*?[:`]\s*(\w+)", text, re.IGNORECASE)
                if regime_match:
                    regimes.append(regime_match.group(1))
                else:
                    regimes.append("unknown")

            conn.execute(
                """INSERT OR REPLACE INTO regime_labels
                   (station, target_date, regimes_active, path_class,
                    phase_summary, review_path, review_source)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    station,
                    target_date,
                    json.dumps(regimes),
                    path_class,
                    json.dumps(phases) if phases else None,
                    str(review_path),
                    "human",
                ),
            )
            count += 1
            log.info("Backfilled regime label for %s from %s", target_date, review_path.name)

        conn.commit()
    finally:
        conn.close()

    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap_single_key(d: dict) -> dict:
    """If dict has exactly one key whose value is a dict or list, unwrap it."""
    if len(d) == 1:
        only_val = next(iter(d.values()))
        if isinstance(only_val, (dict, list)):
            return only_val if isinstance(only_val, dict) else d
    return d


def _find_list(d: dict | list, *keys: str) -> list:
    """Find a list in a dict by trying multiple key names."""
    if isinstance(d, list):
        return d
    for key in keys:
        if key in d and isinstance(d[key], list):
            return d[key]
    # If the dict itself contains list values, return the first one
    for v in d.values():
        if isinstance(v, list):
            return v
    return []


def _looks_like_statistical_audit(d: dict | list) -> bool:
    """Heuristic for identifying the statistical audit YAML block."""
    if not isinstance(d, dict):
        return False
    if "statistical_audit" in d and isinstance(d["statistical_audit"], dict):
        return True
    keys = set(d.keys())
    # Legacy fields or new nested format
    legacy = {"overall_verdict", "promotion_action", "cusum_verdict", "skf_verdict", "dampening_factor"}
    new_nested = {"emos", "boa", "platt", "legacy_shadow"}
    return bool(keys & legacy) or bool(keys & new_nested and "overall_verdict" in keys)


def _extract_yaml_blocks(text: str) -> list[str]:
    """Extract all ```yaml ... ``` fenced code blocks from the text."""
    pattern = re.compile(r"```yaml\s*\n(.*?)```", re.DOTALL)
    return pattern.findall(text)


def _safe_parse_yaml(blocks: list[str], index: int) -> dict | None:
    """Parse a YAML block by index, returning None on failure."""
    if index >= len(blocks):
        return None
    try:
        import yaml
        return yaml.safe_load(blocks[index])
    except Exception:
        log.warning("Failed to parse YAML block %d", index)
        return None


def _extract_narrative(text: str) -> str:
    """Extract the narrative markdown section (after the last YAML block)."""
    # Find the last ``` fence closure
    last_fence = text.rfind("```")
    if last_fence == -1:
        return text

    # Find the next newline after the last fence
    after = text[last_fence + 3:]
    # Look for narrative header (Section 7, 8, or 18 depending on prompt version)
    narrative_match = re.search(
        r"(?:#{1,3}\s*(?:Section\s+\d+|Narrative Review|Full Narrative).*?\n)(.*)",
        after,
        re.DOTALL,
    )
    if narrative_match:
        return narrative_match.group(1).strip()

    # Fallback: everything after the last fence that looks like prose
    stripped = after.strip()
    if len(stripped) > 100:
        return stripped
    return ""


def _models_to_dict(models: list | dict) -> dict:
    """Convert model performance to dict for storage.

    Handles both list-of-model-dicts and structured dict formats.
    """
    if isinstance(models, dict):
        return models
    result = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        key = m.get("model", "unknown")
        source = m.get("source")
        if source:
            key = f"{key}:{source}"
        result[key] = {
            k: v for k, v in m.items()
            if k not in ("model", "source")
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse AI review output and store results")
    sub = parser.add_subparsers(dest="command")

    # Parse a raw review file
    parse_cmd = sub.add_parser("parse", help="Parse raw AI output file")
    parse_cmd.add_argument("--input", required=True, help="Path to raw AI output")
    parse_cmd.add_argument("--db", required=True, help="Path to collector DB")

    # Backfill from human reviews
    backfill_cmd = sub.add_parser("backfill", help="Backfill from human-written reviews")
    backfill_cmd.add_argument("--reviews-dir", required=True, help="Path to reviews/ directory")
    backfill_cmd.add_argument("--db", required=True, help="Path to collector DB")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")

    if args.command == "parse":
        raw = Path(args.input).read_text()
        parsed = parse_review(raw)
        store_review(parsed, args.db)
        print(f"Stored review for {parsed.target_date}: regimes={parsed.regimes_active}")
    elif args.command == "backfill":
        n = backfill_from_human_reviews(Path(args.reviews_dir), args.db)
        print(f"Backfilled {n} reviews")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
