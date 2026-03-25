from __future__ import annotations

from pathlib import Path

import pytest

from analyzer.ai_review_parser import ParsedReview, store_review


def test_store_review_requires_target_date(tmp_path: Path) -> None:
    parsed = ParsedReview(
        station="KMIA",
        target_date="",
        regimes_active=["normal"],
        narrative_markdown="hello",
    )

    with pytest.raises(ValueError, match="missing target_date"):
        store_review(parsed, tmp_path / "miami_collector.db")
