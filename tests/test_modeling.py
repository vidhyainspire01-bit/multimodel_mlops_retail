import pytest

from common.features import FEATURES
from common.modeling import (
    evaluate_scores,
    latest_model_version,
    recency_baseline,
    to_matrix,
    train_lgbm,
)

# skip the whole module when the ML libraries are not installed
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")


def synthetic(n=6000, seed=0):
    """Customers who bought recently and often are more likely to buy again."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["recency_days"] = rng.integers(0, 300, size=n)
    df["purchase_days_90d"] = rng.integers(0, 8, size=n)
    logit = -1.0 - 0.02 * df["recency_days"] + 0.4 * df["purchase_days_90d"]
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def test_to_matrix_orders_columns_and_uses_float64():
    df = synthetic(50)
    df["items_total"] = df["items_total"].round().astype("int64")
    m = to_matrix(df, FEATURES)
    assert list(m.columns) == FEATURES
    assert (m.dtypes == "float64").all()


def test_to_matrix_handles_missing_values():
    df = synthetic(20)
    df["age"] = None
    assert to_matrix(df, FEATURES)["age"].isna().all()


def test_recency_baseline_ranks_recent_buyers_higher():
    df = synthetic()
    assert evaluate_scores(df["label"], recency_baseline(df))["roc_auc"] > 0.6


def test_evaluate_scores_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert evaluate_scores(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert evaluate_scores(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0
    assert evaluate_scores(y, np.array([0.1, 0.2, 0.8, 0.9]))["base_rate"] == 0.5


def test_trained_model_returns_probabilities_and_beats_baseline():
    df = synthetic()
    train, val = df.iloc[:4500], df.iloc[4500:]
    booster = train_lgbm(
        to_matrix(train, FEATURES), train["label"], to_matrix(val, FEATURES), val["label"]
    )
    p = booster.predict(to_matrix(val, FEATURES))
    assert p.min() >= 0.0 and p.max() <= 1.0
    model_auc = evaluate_scores(val["label"], p)["roc_auc"]
    base_auc = evaluate_scores(val["label"], recency_baseline(val))["roc_auc"]
    assert model_auc > 0.65
    assert model_auc >= base_auc - 0.02


def test_training_is_deterministic():
    df = synthetic(3000)
    args = (to_matrix(df, FEATURES), df["label"], to_matrix(df, FEATURES), df["label"])
    p1 = train_lgbm(*args, num_boost_round=30).predict(args[0])
    p2 = train_lgbm(*args, num_boost_round=30).predict(args[0])
    assert np.allclose(p1, p2)


def test_latest_model_version_picks_highest_number():
    class V:
        def __init__(self, v):
            self.version = v

    class Client:
        def search_model_versions(self, query):
            return [V("2"), V("10"), V("9")]

    assert latest_model_version(Client(), "m") == "10"
