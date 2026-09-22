"""Serving-time wrapper: booster + business rules, packaged as one mlflow.pyfunc model.

This is a SEPARATE registered model from the training one (repurchase_lgbm). The training
model's signature is exactly FEATURES (what train.py/evaluate.py score against a holdout
table). The serving model wraps it and additionally accepts customer_id, passing it through
so the caller can match responses to requests, and enriches the output with business rules.
Keeping them separate means the batch pipeline's gate logic (which needs raw probabilities)
never has to know about business rules, and the rules can change without retraining anything.
"""

from common.features import FEATURES
from common.serving_rules import apply_rules

REQUIRED_INPUT_COLUMNS = ["customer_id", *FEATURES]


def make_serving_model(booster):
    """Build an mlflow.pyfunc.PythonModel wrapping a fitted LightGBM booster + business rules.

    Built as a factory (not a module-level class) so importing this module never requires
    mlflow to be installed - only calling this function does, matching common/ts_modeling.py.
    """
    import mlflow.pyfunc
    import pandas as pd

    class ServingModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: pd.DataFrame):
            missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in model_input.columns]
            if missing:
                raise ValueError(f"missing required input columns: {missing}")

            X = model_input[FEATURES].astype("float64")
            raw_probs = booster.predict(X)

            rows = [
                {
                    "customer_id": customer_id,
                    **apply_rules(prob, recency, tenure),
                }
                for customer_id, prob, recency, tenure in zip(
                    model_input["customer_id"],
                    raw_probs,
                    model_input["recency_days"],
                    model_input["tenure_days"],
                )
            ]
            return pd.DataFrame(rows)

    return ServingModel()
