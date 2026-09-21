# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / evaluate - champion vs challenger gate
# MAGIC Scores the challenger and the current champion on the SAME holdout snapshot.
# MAGIC The challenger must also beat a naive recency baseline to be promoted at all.
# MAGIC * beats champion by > 0.002 AUC -> promoted (`champion` alias moves)
# MAGIC * no meaningful difference -> champion kept, task succeeds
# MAGIC * worse than champion (> 0.005 AUC) or below baseline -> not promoted; task fails when
# MAGIC   `fail_on_regression=true` (staging), stays green in prod

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("challenger_version", "")
dbutils.widgets.text("fail_on_regression", "false")

catalog = dbutils.widgets.get("catalog")
fail_on_regression = dbutils.widgets.get("fail_on_regression").lower() == "true"
model_name = f"{catalog}.repurchase.repurchase_lgbm"

# COMMAND ----------

# Make the bundle root importable (notebooks run from notebooks/repurchase/).
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

import mlflow
from mlflow.tracking import MlflowClient

from common.features import FEATURES, LABEL
from common.modeling import evaluate_scores, latest_model_version, recency_baseline, to_matrix
from common.promotion import promote_if_better

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

challenger_version = dbutils.widgets.get("challenger_version") or latest_model_version(
    MlflowClient(), model_name
)

holdout = spark.table(f"{catalog}.repurchase.holdout").toPandas()
X, y = to_matrix(holdout, FEATURES), holdout[LABEL]
baseline_auc = evaluate_scores(y, recency_baseline(holdout))["roc_auc"]
print(f"holdout rows={len(holdout):,}, recency-baseline AUC={baseline_auc:.4f}")


def score_fn(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    metrics = evaluate_scores(y, model.predict(X))
    print(f"{model_uri}: AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f}")
    return metrics["roc_auc"]


# COMMAND ----------

status = promote_if_better(
    model_name,
    challenger_version,
    score_fn,
    higher_is_better=True,
    min_delta=0.002,
    absolute_threshold=baseline_auc,   # must beat the naive recency baseline
    fail_on_regression=fail_on_regression,
    regression_tolerance=0.005,
)
print(f"gate outcome for v{challenger_version}: {status}")
dbutils.jobs.taskValues.set("promotion_status", status)
