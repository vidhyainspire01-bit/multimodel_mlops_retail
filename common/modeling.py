"""Model helpers shared by train / evaluate / batch_score notebooks."""

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def to_matrix(pdf, features):
    """Feature matrix as float64 in a fixed column order (same for train, evaluate, score)."""
    return pdf[features].astype("float64")


def recency_baseline(pdf):
    """Naive score: the more recently someone bought, the more likely they buy again."""
    return -pdf["recency_days"].astype("float64")


def evaluate_scores(y_true, y_score):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
    }


def train_lgbm(X_train, y_train, X_val, y_val, params=None, num_boost_round=1000, patience=50):
    """Native LightGBM Booster, so the MLflow pyfunc `predict` returns probabilities."""
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    return lgb.train(
        p,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )


def latest_model_version(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for {model_name}")
    return str(max(int(v.version) for v in versions))
