# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / train
# MAGIC Trains LightGBM on the earlier snapshots, early-stops on the most recent training snapshot
# MAGIC (its label window ends before the holdout's begins), and registers the model in Unity Catalog.
# MAGIC The registered version is published as the task value `challenger_version`.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
catalog = dbutils.widgets.get("catalog")
model_name = f"{catalog}.repurchase.repurchase_lgbm"

# COMMAND ----------

# Make the bundle root importable (notebooks run from notebooks/repurchase/).
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from common.features import FEATURES, LABEL
from common.modeling import (
    DEFAULT_PARAMS,
    evaluate_scores,
    latest_model_version,
    recency_baseline,
    to_matrix,
    train_lgbm,
)

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

pdf = spark.table(f"{catalog}.repurchase.train_features").toPandas()
last_snapshot = pdf["snapshot_date"].max()
train_df = pdf[pdf["snapshot_date"] < last_snapshot]
val_df = pdf[pdf["snapshot_date"] == last_snapshot]

X_train, y_train = to_matrix(train_df, FEATURES), train_df[LABEL]
X_val, y_val = to_matrix(val_df, FEATURES), val_df[LABEL]
print(f"train rows={len(train_df):,}, validation rows={len(val_df):,} (snapshot {last_snapshot})")

# COMMAND ----------

with mlflow.start_run(run_name="repurchase-lgbm") as run:
    booster = train_lgbm(X_train, y_train, X_val, y_val)

    val_pred = booster.predict(X_val)
    val_metrics = evaluate_scores(y_val, val_pred)
    baseline_metrics = evaluate_scores(y_val, recency_baseline(val_df))

    mlflow.log_params({**DEFAULT_PARAMS, "best_iteration": booster.best_iteration})
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_metrics(
        {
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_baseline_roc_auc": baseline_metrics["roc_auc"],
            "n_train_rows": float(len(train_df)),
        }
    )

    sample = X_val.head(500)
    signature = infer_signature(sample, booster.predict(sample))
    info = mlflow.lightgbm.log_model(
        lgb_model=booster,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name,
    )

version = getattr(info, "registered_model_version", None) or latest_model_version(
    MlflowClient(), model_name
)
print(f"Registered {model_name} version {version}")
print(f"validation AUC {val_metrics['roc_auc']:.4f} vs recency baseline {baseline_metrics['roc_auc']:.4f}")

assert val_metrics["roc_auc"] > 0.5, "model is no better than random on validation"

dbutils.jobs.taskValues.set("challenger_version", str(version))
