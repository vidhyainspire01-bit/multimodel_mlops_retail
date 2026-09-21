# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / assert_outputs
# MAGIC Integration checks on the scored table and the registry. Any failed assertion fails the
# MAGIC task, which fails the job, which blocks promotion to the next environment.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
catalog = dbutils.widgets.get("catalog")
model_name = f"{catalog}.repurchase.repurchase_lgbm"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")

scores = spark.table(f"{catalog}.repurchase.scores")
expected = spark.table(f"{catalog}.repurchase.score_features")

n_scores = scores.count()
n_expected = expected.count()
n_distinct = scores.select("customer_id").distinct().count()
n_null = scores.filter(F.col("repurchase_probability").isNull()).count()
n_out_of_range = scores.filter(
    (F.col("repurchase_probability") < 0) | (F.col("repurchase_probability") > 1)
).count()
mean_p, std_p = scores.agg(
    F.avg("repurchase_probability"), F.stddev("repurchase_probability")
).first()
versions = [r.model_version for r in scores.select("model_version").distinct().collect()]
champion = MlflowClient().get_model_version_by_alias(model_name, "champion")

print(f"scores={n_scores:,} expected={n_expected:,} mean_p={mean_p:.3f} std_p={std_p:.3f}")
print(f"scored with versions {versions}; current champion v{champion.version}")

# COMMAND ----------

assert n_scores > 0, "scores table is empty"
assert n_scores == n_expected, f"scored {n_scores} customers but expected {n_expected}"
assert n_distinct == n_scores, "duplicate customers in scores"
assert n_null == 0, f"{n_null} null probabilities"
assert n_out_of_range == 0, f"{n_out_of_range} probabilities outside [0, 1]"
assert 0.01 < mean_p < 0.7, f"mean probability {mean_p:.3f} looks implausible"
assert std_p > 0.01, f"scores are nearly constant (std {std_p:.4f}); model is not discriminating"
assert versions == [str(champion.version)], "scores were not produced by the current champion"
print("All output assertions passed.")
