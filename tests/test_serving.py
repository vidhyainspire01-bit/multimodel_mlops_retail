import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("lightgbm")
pytest.importorskip("mlflow")

import lightgbm as lgb

from common.features import FEATURES
from common.serving import REQUIRED_INPUT_COLUMNS, make_serving_model


def toy_booster():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f: rng.normal(size=200) for f in FEATURES})
    y = (X["recency_days"] < 0).astype(int)   # arbitrary learnable pattern
    return lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), 20)


def sample_input(n=4):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    df["customer_id"] = [f"c{i}" for i in range(n)]
    df["recency_days"] = [1.0, 40.0, 2.0, 100.0]
    df["tenure_days"] = [300.0, 5.0, 300.0, 300.0]
    return df


def test_required_columns_include_customer_id_and_all_features():
    assert REQUIRED_INPUT_COLUMNS[0] == "customer_id"
    assert set(REQUIRED_INPUT_COLUMNS[1:]) == set(FEATURES)


def test_predict_returns_one_row_per_input_row_with_expected_columns():
    model = make_serving_model(toy_booster())
    out = model.predict(None, sample_input(4))
    assert len(out) == 4
    assert set(out.columns) == {
        "customer_id", "repurchase_probability", "risk_tier", "recommended_action", "low_confidence",
    }


def test_predict_preserves_customer_id_order():
    model = make_serving_model(toy_booster())
    inp = sample_input(4)
    out = model.predict(None, inp)
    assert list(out["customer_id"]) == list(inp["customer_id"])


def test_predict_raises_clear_error_on_missing_column():
    model = make_serving_model(toy_booster())
    bad_input = sample_input(4).drop(columns=["recency_days"])
    with pytest.raises(ValueError, match="missing required input columns"):
        model.predict(None, bad_input)


def test_predict_probabilities_are_valid_range():
    model = make_serving_model(toy_booster())
    out = model.predict(None, sample_input(4))
    assert out["repurchase_probability"].between(0, 1).all()


def test_low_tenure_customer_flagged_low_confidence_end_to_end():
    model = make_serving_model(toy_booster())
    out = model.predict(None, sample_input(4))
    # sample row index 1 has tenure_days=5.0, below the confidence threshold
    assert out.loc[out["customer_id"] == "c1", "low_confidence"].iloc[0] == True
