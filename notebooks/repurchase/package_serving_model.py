# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / package_serving_model
# MAGIC Wraps the current `@champion` training model (`repurchase_lgbm`) with the business rules
# MAGIC in `common/serving_rules.py` and registers the result as a SEPARATE model,
# MAGIC `repurchase_lgbm_serving`. This is the model you point "Serve this model" / a Model
# MAGIC Serving endpoint at - it accepts `customer_id` + the 16 features and returns
# MAGIC risk_tier / recommended_action alongside the raw probability.
# MAGIC
# MAGIC Runs after `assert_outputs` so a serving model is only packaged once the batch pipeline
# MAGIC has confirmed the champion is healthy on this run.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3,<5" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
catalog = dbutils.widgets.get("catalog")
train_model_name = f"{catalog}.repurchase.repurchase_lgbm"
serve_model_name = f"{catalog}.repurchase.repurchase_lgbm_serving"

# COMMAND ----------

# Make the bundle root importable (notebooks run from notebooks/repurchase/).
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from common.serving import REQUIRED_INPUT_COLUMNS, make_serving_model

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

client = MlflowClient()
champion = client.get_model_version_by_alias(train_model_name, "champion")
booster = mlflow.lightgbm.load_model(f"models:/{train_model_name}@champion")
print(f"wrapping {train_model_name} v{champion.version} ({booster.num_feature()} features)")

serving_model = make_serving_model(booster)

# a tiny example row so the logged signature matches exactly what a serving request looks like
example = pd.DataFrame([{c: 0.0 for c in REQUIRED_INPUT_COLUMNS}])
example["customer_id"] = "example-customer"
example_out = serving_model.predict(None, example)
signature = infer_signature(example, example_out)

with mlflow.start_run(run_name="package-serving-model"):
    mlflow.log_param("wraps_training_model", train_model_name)
    mlflow.log_param("wraps_training_version", champion.version)
    info = mlflow.pyfunc.log_model(
        python_model=serving_model,
        artifact_path="model",
        signature=signature,
        input_example=example,
        registered_model_name=serve_model_name,
    )

serve_version = str(info.registered_model_version)
client.set_registered_model_alias(serve_model_name, "champion", serve_version)
print(f"Registered {serve_model_name} v{serve_version}, pointing at @champion")
print("This is the model to open in Catalog Explorer and click 'Serve this model' on.")
