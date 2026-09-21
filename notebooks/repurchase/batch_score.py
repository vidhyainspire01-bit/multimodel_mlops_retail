# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / batch_score
# MAGIC Scores every customer in `score_features` with the `@champion` model and writes
# MAGIC `<catalog>.repurchase.scores`. If the challenger lost the gate, the existing champion scores.

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
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from common.features import FEATURES
from common.modeling import to_matrix

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

client = MlflowClient()
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
except MlflowException as e:
    raise RuntimeError(
        f"No @champion for {model_name}. The evaluate task promotes the first passing model; "
        "if it was rejected there is nothing to score with."
    ) from e

model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

feats = spark.table(f"{catalog}.repurchase.score_features").toPandas()
out = feats[["customer_id", "snapshot_date"]].copy()
out["repurchase_probability"] = model.predict(to_matrix(feats, FEATURES))

scores = (
    spark.createDataFrame(out)
    .withColumn("model_version", F.lit(str(champion.version)))
    .withColumn("scored_at", F.current_timestamp())
)
(
    scores.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.repurchase.scores")
)
print(f"Scored {len(out):,} customers with {model_name} v{champion.version}")
