"""Project entrypoints that wrap the older runner with new maintenance hooks."""

from __future__ import annotations

import logging

from collector.prune import default_db_path, prune_database
from collector.runner import cli_entry as _runner_cli_entry

log = logging.getLogger(__name__)


def _load_retention_overrides() -> dict[str, int]:
    """Load retention overrides from config.toml [retention] section."""
    try:
        import tomllib
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[2] / "config.toml"
        if not config_path.exists():
            return {}
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        retention = cfg.get("retention", {})
        return {k: int(v) for k, v in retention.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def cli_entry() -> int:
    """Prune the DB on startup, then delegate to the existing collector runner."""
    try:
        overrides = _load_retention_overrides()
        counts = prune_database(default_db_path(), retention_overrides=overrides)
        total = sum(counts.values())
        if total > 0:
            log.info("Startup prune: %d rows removed", total)
    except Exception:
        # Pruning should not block data collection startup.
        pass
    return _runner_cli_entry()
