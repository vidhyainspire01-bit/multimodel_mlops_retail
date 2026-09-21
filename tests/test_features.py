from datetime import date, timedelta

import pytest

from common.features import FEATURES, KEYS, LABEL, check_no_leakage, make_snapshots

MAX = date(2020, 9, 22)


def test_holdout_label_window_ends_exactly_at_max_date():
    snap = make_snapshots(MAX, 28, 4)
    assert snap["holdout"] + timedelta(days=28) == MAX
    assert snap["score"] == MAX


def test_expected_dates_for_hm_range():
    snap = make_snapshots(MAX, 28, 4)
    assert snap["holdout"] == date(2020, 8, 25)
    assert snap["train"] == [
        date(2020, 7, 28),
        date(2020, 6, 30),
        date(2020, 6, 2),
        date(2020, 5, 5),
    ]


def test_train_label_windows_end_before_holdout_snapshot():
    snap = make_snapshots(MAX, 28, 6)
    assert all(s + timedelta(days=28) <= snap["holdout"] for s in snap["train"])


def test_generated_snapshots_pass_leakage_check():
    check_no_leakage(make_snapshots(MAX, 28, 4), MAX, 28)
    check_no_leakage(make_snapshots(MAX, 14, 8), MAX, 14)


def test_leakage_check_detects_train_overlapping_holdout():
    snap = make_snapshots(MAX, 28, 4)
    snap["train"][0] = snap["holdout"] - timedelta(days=5)
    with pytest.raises(ValueError):
        check_no_leakage(snap, MAX, 28)


def test_leakage_check_detects_overlapping_train_windows():
    snap = make_snapshots(MAX, 28, 3)
    snap["train"][1] = snap["train"][0] - timedelta(days=10)
    with pytest.raises(ValueError):
        check_no_leakage(snap, MAX, 28)


def test_leakage_check_detects_holdout_beyond_data():
    snap = make_snapshots(MAX, 28, 2)
    snap["holdout"] = MAX - timedelta(days=3)
    with pytest.raises(ValueError):
        check_no_leakage(snap, MAX, 28)


@pytest.mark.parametrize("h,n", [(0, 3), (28, 0), (-1, 1)])
def test_invalid_arguments_rejected(h, n):
    with pytest.raises(ValueError):
        make_snapshots(MAX, h, n)


def test_feature_list_is_clean():
    assert len(FEATURES) == len(set(FEATURES))
    assert LABEL not in FEATURES
    assert not set(KEYS) & set(FEATURES)
