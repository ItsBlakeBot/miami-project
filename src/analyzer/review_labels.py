"""Extract structured labels from canonical daily reviews.

The reviews remain useful in the hybrid workflow, but as label sources and case
annotations — not as the only thing the live bot reasons from.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUMMARY_FIELD_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:?\s*(.+?)\s*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

CLAMP_LABELS = {
    "high_locked_early",
    "high_bias_locked_in",
    "warm_low_brackets_dead",
    "low_endpoint_locked",
}


@dataclass
class ReviewLabelRecord:
    target_date: str
    station: str | None = None
    time_basis: str | None = None
    climate_day: str | None = None
    primary_regime: str | None = None
    background_regime: str | None = None
    path_class: str | None = None
    core_day_type: str | None = None
    cli_high_f: float | None = None
    cli_low_f: float | None = None
    obs_high_f: float | None = None
    obs_low_f: float | None = None
    signal_labels: list[str] = field(default_factory=list)
    clamp_labels: list[str] = field(default_factory=list)
    best_high_expression: str | None = None
    best_low_expression: str | None = None
    source_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_review_labels(markdown: str, *, source_path: str | None = None) -> ReviewLabelRecord:
    target_date = _extract_target_date(markdown, source_path)
    summary = _extract_structured_summary(markdown)
    signal_labels = sorted(set(_extract_signal_labels(markdown)))
    clamp_labels = sorted(label for label in signal_labels if label in CLAMP_LABELS)

    return ReviewLabelRecord(
        target_date=target_date,
        station=summary.get("Station"),
        time_basis=summary.get("Time basis"),
        climate_day=summary.get("Climate day"),
        primary_regime=_strip_ticks(summary.get("Primary regime")),
        background_regime=_strip_ticks(summary.get("Background regime")),
        path_class=_strip_ticks(summary.get("Path class")),
        core_day_type=summary.get("Core day type"),
        cli_high_f=_extract_first_float(summary.get("CLI settlement high")),
        cli_low_f=_extract_first_float(summary.get("CLI settlement low")),
        obs_high_f=_extract_first_float(summary.get("Obs high")),
        obs_low_f=_extract_first_float(summary.get("Obs low")),
        signal_labels=signal_labels,
        clamp_labels=clamp_labels,
        best_high_expression=_extract_best_expression(markdown, "high"),
        best_low_expression=_extract_best_expression(markdown, "low"),
        source_path=source_path,
    )


def load_review_label_map(reviews_dir: Path) -> dict[str, ReviewLabelRecord]:
    records: dict[str, ReviewLabelRecord] = {}
    for path in sorted(reviews_dir.glob("*.md")):
        record = extract_review_labels(path.read_text(), source_path=str(path))
        records[record.target_date] = record
    return records


def _extract_target_date(markdown: str, source_path: str | None) -> str:
    header_match = re.search(r"^#\s+KMIA Daily Review\s+—\s+(\d{4}-\d{2}-\d{2})", markdown, re.M)
    if header_match:
        return header_match.group(1)
    if source_path:
        path_match = re.search(r"(\d{4}-\d{2}-\d{2})", source_path)
        if path_match:
            return path_match.group(1)
    raise ValueError("Unable to determine target date from review")


def _extract_structured_summary(markdown: str) -> dict[str, str]:
    start = markdown.find("## Structured Summary")
    if start == -1:
        return {}
    tail = markdown[start:].split("\n## ", 1)[0]
    result: dict[str, str] = {}
    for line in tail.splitlines():
        match = SUMMARY_FIELD_RE.match(line.strip())
        if match:
            key = match.group(1).rstrip(":")
            result[key] = match.group(2)
    return result


def _extract_signal_labels(markdown: str) -> list[str]:
    labels: list[str] = []
    sections = []
    for heading in ["## Signal-to-Adjustment Notes", "## Training Retrieval Notes"]:
        idx = markdown.find(heading)
        if idx != -1:
            sections.append(markdown[idx:].split("\n## ", 1)[0])
    for section in sections:
        labels.extend(BACKTICK_RE.findall(section))
    return labels


def _extract_best_expression(markdown: str, side: str) -> str | None:
    pattern = re.compile(
        rf"### Best {re.escape(side)}-side idea.*?\n-\s+\*\*(.+?)\*\*",
        re.S,
    )
    match = pattern.search(markdown)
    if match:
        return match.group(1).strip()
    retrieval_pattern = re.compile(
        rf"Best {re.escape(side)}-side expression:\s+(.+)",
        re.I,
    )
    match = retrieval_pattern.search(markdown)
    if match:
        return match.group(1).strip()
    return None


def _extract_first_float(value: str | None) -> float | None:
    if not value:
        return None
    match = FLOAT_RE.search(value)
    return float(match.group(1)) if match else None


def _strip_ticks(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip("`")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract structured labels from daily review markdown files.")
    parser.add_argument("input", help="Review file or reviews directory")
    parser.add_argument("--output", help="Output NDJSON path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.md"))
    else:
        paths = [input_path]

    records = [extract_review_labels(path.read_text(), source_path=str(path)).to_dict() for path in paths]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    else:
        for record in records:
            print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
