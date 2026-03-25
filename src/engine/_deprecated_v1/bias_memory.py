"""Slow bias-memory layer.

This is the structural-bias component of the hybrid architecture.
It should react slowly and never be confused with same-day event interpretation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BiasRecord:
    bias_f: float = 0.0
    sample_count: int = 0
    alpha: float = 0.15

    def update(self, error_f: float | None) -> None:
        if error_f is None:
            return
        self.bias_f = (1.0 - self.alpha) * self.bias_f + self.alpha * error_f
        self.sample_count += 1

    def corrected(self, value_f: float | None) -> float | None:
        if value_f is None:
            return None
        return value_f - self.bias_f


class JsonBiasMemory:
    """Tiny JSON-backed store for slow source bias.

    This is intentionally simple for v1. The store can later move into SQLite if
    that proves more convenient, but JSON is easy to inspect and debug.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, BiasRecord] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._records = {}
            return
        raw = json.loads(self.path.read_text())
        self._records = {key: BiasRecord(**value) for key, value in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self._records.items()}, indent=2, sort_keys=True))

    def get(self, key: str, *, alpha: float = 0.15) -> BiasRecord:
        record = self._records.get(key)
        if record is None:
            record = BiasRecord(alpha=alpha)
            self._records[key] = record
        return record

    def update(self, key: str, error_f: float | None, *, alpha: float = 0.15) -> BiasRecord:
        record = self.get(key, alpha=alpha)
        record.update(error_f)
        return record

    def corrected(self, key: str, value_f: float | None, *, alpha: float = 0.15) -> float | None:
        return self.get(key, alpha=alpha).corrected(value_f)


def bias_key(*, station: str, family: str, market_type: str, checkpoint_bucket: str = "all") -> str:
    return f"{station}|{family}|{market_type}|{checkpoint_bucket}"
