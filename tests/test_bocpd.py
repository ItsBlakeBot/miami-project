from __future__ import annotations

import numpy as np

from engine.bocpd import BOCPDConfig, GaussianMeanBOCPD


def test_bocpd_outputs_valid_probabilities_on_shifted_stream() -> None:
    rng = np.random.default_rng(42)
    model = GaussianMeanBOCPD(
        BOCPDConfig(
            hazard=0.05,
            max_run_length=100,
            obs_variance=1.0,
            mu0=0.0,
            kappa0=1.0,
        )
    )

    baseline = rng.normal(0.0, 0.25, size=35)
    shifted = rng.normal(2.2, 0.25, size=8)

    probs = []
    for x in np.concatenate([baseline, shifted]):
        p = model.update(float(x))
        probs.append(p)

    assert len(probs) == 43
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(float(np.sum(model.run_length_posterior)) - 1.0) < 1e-9


def test_bocpd_reset_restores_initial_state() -> None:
    model = GaussianMeanBOCPD(BOCPDConfig(max_run_length=20))
    _ = model.update(0.5)
    _ = model.update(-0.2)

    model.reset()
    posterior = model.run_length_posterior

    assert posterior[0] == 1.0
    assert abs(float(np.sum(posterior)) - 1.0) < 1e-9
