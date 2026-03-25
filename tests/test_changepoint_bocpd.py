from __future__ import annotations

from engine.changepoint_detector import CHANNELS, ChangeDetector


def _disable_cusum_layer(detector: ChangeDetector) -> None:
    for ch in CHANNELS:
        if ch in detector._cusum:
            detector._cusum[ch].h = 1e9
            detector._cusum[ch].k = 1e6


def test_change_detector_can_fire_bocpd_layer() -> None:
    detector = ChangeDetector(use_bocpd=True, bocpd_fire_threshold=0.01)

    # Fit simple diurnal baseline for temp channel.
    xs = [0.0, 6.0, 12.0, 18.0, 23.0]
    ys = [70.0, 70.0, 70.0, 70.0, 70.0]
    detector._diurnal.fit("temp_f", xs, ys)

    detector.reset()
    _disable_cusum_layer(detector)

    fired_layer3 = False

    # Stable segment
    for _ in range(18):
        state = detector.update({"temp_f": 70.0}, hour_lst=13.0, minutes_elapsed=5.0)
        assert state.layer in (0, 3)
        assert state.bocpd_run_length_mode is not None
        assert state.bocpd_run_length_expectation is not None

    # Mean-shift segment: gradual enough to avoid threshold spikes, but persistent.
    for _ in range(6):
        state = detector.update({"temp_f": 71.4}, hour_lst=13.2, minutes_elapsed=5.0)
        if state.fired and state.layer == 3:
            fired_layer3 = True
            assert state.minutes_since_last_changepoint == 0.0
            break

    assert fired_layer3, "Expected BOCPD layer to fire after persistent mean shift"


def test_change_detector_minutes_since_last_changepoint_accumulates_without_fire() -> None:
    detector = ChangeDetector(use_bocpd=True, bocpd_fire_threshold=0.99)

    xs = [0.0, 6.0, 12.0, 18.0, 23.0]
    ys = [70.0, 70.0, 70.0, 70.0, 70.0]
    detector._diurnal.fit("temp_f", xs, ys)

    detector.reset()
    _disable_cusum_layer(detector)

    state1 = detector.update({"temp_f": 70.0}, hour_lst=13.0, minutes_elapsed=5.0)
    state2 = detector.update({"temp_f": 70.1}, hour_lst=13.1, minutes_elapsed=5.0)
    state3 = detector.update({"temp_f": 70.0}, hour_lst=13.2, minutes_elapsed=10.0)

    assert state1.fired is False
    assert state2.fired is False
    assert state3.fired is False
    assert state1.minutes_since_last_changepoint == 5.0
    assert state2.minutes_since_last_changepoint == 10.0
    assert state3.minutes_since_last_changepoint == 20.0
