from __future__ import annotations

from engine.emos import EMOSCoefficients, EMOSState
from engine.orchestrator import _resolve_emos_coefficients


def test_resolve_emos_coefficients_uses_state_when_present() -> None:
    coeff = EMOSCoefficients(a=0.2, b=0.9, c=0.5, d=0.3, market_type="high")
    state = EMOSState(high=coeff, low=None)
    notes: list[str] = []

    out = _resolve_emos_coefficients(state, "high", notes)

    assert out is coeff
    assert notes == []


def test_resolve_emos_coefficients_falls_back_to_identity() -> None:
    state = EMOSState(high=None, low=None)
    notes: list[str] = []

    out = _resolve_emos_coefficients(state, "low", notes)

    assert out.a == 0.0
    assert out.b == 1.0
    assert out.c == 0.0
    assert out.d == 1.0
    assert out.market_type == "low"
    assert out.n_training_samples == 0
    assert out.fit_utc == "cold_start_identity"
    assert any("cold-start identity" in n for n in notes)
