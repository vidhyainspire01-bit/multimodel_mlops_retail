import sys
import types

import pytest

from common.promotion import (
    KEPT,
    PROMOTED,
    REJECTED,
    decide,
    is_regression,
    meets_threshold,
    promote_if_better,
)

# ---------- pure decision logic ----------


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


def test_regression_needs_to_exceed_tolerance():
    assert is_regression(0.80, 0.79, True, tolerance=0.005)
    assert not is_regression(0.80, 0.798, True, tolerance=0.005)
    assert is_regression(0.30, 0.32, False, tolerance=0.01)
    assert not is_regression(0.30, 0.305, False, tolerance=0.01)


# ---------- full gate flow against a fake registry ----------


@pytest.fixture
def registry(monkeypatch):
    aliases = {}

    class FakeMlflowException(Exception):
        pass

    class FakeVersion:
        def __init__(self, version):
            self.version = version

    class FakeClient:
        def set_registered_model_alias(self, name, alias, version):
            aliases[(name, alias)] = str(version)

        def get_model_version_by_alias(self, name, alias):
            if (name, alias) not in aliases:
                raise FakeMlflowException("alias not found")
            return FakeVersion(aliases[(name, alias)])

    mlflow = types.ModuleType("mlflow")
    mlflow.set_registry_uri = lambda uri: None
    exceptions = types.ModuleType("mlflow.exceptions")
    exceptions.MlflowException = FakeMlflowException
    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = FakeClient
    mlflow.exceptions, mlflow.tracking = exceptions, tracking
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    return aliases


MODEL = "dev_ml.repurchase.repurchase_lgbm"


def scorer(by_version):
    return lambda uri: by_version[uri.rsplit("/", 1)[1]]


def test_first_model_becomes_champion(registry):
    out = promote_if_better(MODEL, 1, scorer({"1": 0.75}), True)
    assert out == PROMOTED
    assert registry[(MODEL, "champion")] == "1"


def test_better_challenger_takes_champion_alias(registry):
    registry[(MODEL, "champion")] = "1"
    out = promote_if_better(MODEL, 2, scorer({"1": 0.75, "2": 0.78}), True, min_delta=0.002)
    assert out == PROMOTED
    assert registry[(MODEL, "champion")] == "2"


def test_tie_keeps_champion_and_does_not_fail_even_when_strict(registry):
    registry[(MODEL, "champion")] = "1"
    out = promote_if_better(
        MODEL, 2, scorer({"1": 0.75, "2": 0.7505}), True,
        min_delta=0.002, fail_on_regression=True,
    )
    assert out == KEPT
    assert registry[(MODEL, "champion")] == "1"


def test_worse_challenger_is_not_promoted(registry):
    registry[(MODEL, "champion")] = "1"
    out = promote_if_better(MODEL, 2, scorer({"1": 0.75, "2": 0.65}), True)
    assert out == REJECTED
    assert registry[(MODEL, "champion")] == "1"      # champion untouched


def test_worse_challenger_fails_the_task_when_strict(registry):
    registry[(MODEL, "champion")] = "1"
    with pytest.raises(RuntimeError):
        promote_if_better(
            MODEL, 2, scorer({"1": 0.75, "2": 0.65}), True, fail_on_regression=True
        )
    assert registry[(MODEL, "champion")] == "1"


def test_model_below_baseline_is_never_promoted(registry):
    out = promote_if_better(MODEL, 1, scorer({"1": 0.60}), True, absolute_threshold=0.70)
    assert out == REJECTED
    assert (MODEL, "champion") not in registry


def test_below_baseline_fails_when_strict(registry):
    with pytest.raises(RuntimeError):
        promote_if_better(
            MODEL, 1, scorer({"1": 0.60}), True,
            absolute_threshold=0.70, fail_on_regression=True,
        )


def test_lower_is_better_metric_promotes_lower_error(registry):
    registry[(MODEL, "champion")] = "1"
    out = promote_if_better(MODEL, 2, scorer({"1": 0.30, "2": 0.25}), False, min_delta=0.01)
    assert out == PROMOTED
    assert registry[(MODEL, "champion")] == "2"
