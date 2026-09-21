from common.promotion import decide, meets_threshold


def test_higher_is_better_promotes_when_clearly_better():
    assert decide(0.80, 0.83, higher_is_better=True, min_delta=0.002)


def test_higher_is_better_rejects_worse():
    assert not decide(0.80, 0.79, higher_is_better=True)


def test_min_delta_blocks_noise():
    assert not decide(0.800, 0.801, higher_is_better=True, min_delta=0.002)


def test_tie_is_not_promoted():
    assert not decide(0.80, 0.80, higher_is_better=True)


def test_lower_is_better_promotes_lower_error():
    assert decide(0.30, 0.25, higher_is_better=False, min_delta=0.01)


def test_lower_is_better_rejects_higher_error():
    assert not decide(0.30, 0.35, higher_is_better=False)


def test_threshold_floor_for_score_metrics():
    assert meets_threshold(0.65, True, 0.6)
    assert not meets_threshold(0.55, True, 0.6)


def test_threshold_ceiling_for_error_metrics():
    assert meets_threshold(0.25, False, 0.3)
    assert not meets_threshold(0.35, False, 0.3)


def test_threshold_disabled_when_none():
    assert meets_threshold(0.0, True, None)
