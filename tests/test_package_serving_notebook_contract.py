"""Not a Databricks run - just locks the input/output contract the packaging notebook promises,
so a change to common/serving.py or common/features.py that breaks the serving payload shape
fails fast in CI instead of on the next 'Serve this model' click.
"""

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("mlflow")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import lightgbm as lgb

from common.serving import REQUIRED_INPUT_COLUMNS, make_serving_model


def test_example_row_round_trips_like_the_notebook_does():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=50) for c in REQUIRED_INPUT_COLUMNS if c != "customer_id"})
    y = (X["recency_days"] < 0).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), 10)

    example = pd.DataFrame([{c: 0.0 for c in REQUIRED_INPUT_COLUMNS}])
    example["customer_id"] = "example-customer"

    out = make_serving_model(booster).predict(None, example)
    assert list(out.columns) == [
        "customer_id", "repurchase_probability", "risk_tier", "recommended_action", "low_confidence",
    ]
    assert out["customer_id"].iloc[0] == "example-customer"
