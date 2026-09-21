# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / train  (STUB)

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
catalog = dbutils.widgets.get("catalog")
print(f"train stub running against {catalog}")

# COMMAND ----------

# Real version: log + register the model, then publish the registered version number.
dbutils.jobs.taskValues.set("challenger_version", "0")
